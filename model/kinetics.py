import numpy as np


def compute_Xv(TCD, viability):
    """Расчёт концентрации жизнеспособных клеток"""
    return TCD * viability


def compute_viability(Xv, TCD):
    """Расчёт жизнеспособности"""
    return Xv / TCD if TCD > 0 else 0.0

"""
def mu_growth(G, Lac, NH4, p, T_phase, temp_coeffs):
    #Удельная скорость роста с температурной коррекцией
    if T_phase == 0:  # 37°C
        mu_max = p["mu_max"]
    else:  # 33°C
        mu_max = p["mu_max"] * temp_coeffs["mu_max_factor"]

    # Добавляем ограничители, чтобы избежать отрицательных значений
    mu_value = (
            mu_max
            * (G / (p["K_s"] + G + 1e-6))
            * (p["Ki_L"] / (p["Ki_L"] + Lac + 1e-6))
            * (p["Ki_N"] / (p["Ki_N"] + NH4 + 1e-6))
    )
    return max(mu_value, 0.0)  # Не может быть отрицательным
"""

def mu_growth(G, Lac, NH4, p, T_phase, temp_coeffs, t=None, t_shift=None):
    """
    Удельная скорость роста с температурной коррекцией
    Добавлено прогрессивное снижение после сдвига
    """
    # Базовый mu_max
    if T_phase == 0:  # 37°C
        mu_max = p["mu_max"]
    else:  # 33°C
        mu_max_base = p["mu_max"] * temp_coeffs["mu_max_factor"]

        # Дополнительное снижение со временем после сдвига
        if t is not None and t_shift is not None:
            hours_post_shift = t - t_shift
            if hours_post_shift > 0:
                # Прогрессивное снижение: каждые 24 часа после сдвига - еще 10%
                time_factor = 1.0 - min(0.5, hours_post_shift / 240 * 0.5)
                mu_max = mu_max_base * time_factor
            else:
                mu_max = mu_max_base
        else:
            mu_max = mu_max_base

    # Ингибирование метаболитами (можно усилить после сдвига)
    if T_phase == 1:  # После сдвига
        Ki_L = p.get("Ki_L_post", p["Ki_L"] * 0.8)  # Усилить ингибирование лактатом
        Ki_N = p.get("Ki_N_post", p["Ki_N"] * 0.8)  # Усилить ингибирование аммонием
    else:
        Ki_L = p["Ki_L"]
        Ki_N = p["Ki_N"]

    # Добавляем ограничители, чтобы избежать отрицательных значений
    mu_value = (
            mu_max
            * (G / (p["K_s"] + G + 1e-6))
            * (Ki_L / (Ki_L + Lac + 1e-6))
            * (Ki_N / (Ki_N + NH4 + 1e-6))
    )

    return max(mu_value, 0.0)


def q_glucose(mu, p, T_phase, temp_coeffs):
    """Удельная скорость потребления глюкозы"""
    if T_phase == 0:  # 37°C
        Y_XG = p["Y_XG"]
        m_G = p["m_G"]
    else:  # 33°C

        Y_XG = p["Y_XG"] * 0.9  #
        m_G = p["m_G"] * 0.8

    return mu / (Y_XG + 1e-6) + m_G


def q_lactate(qG, p, T_phase, temp_coeffs):
    """Удельная скорость образования лактата"""
    if T_phase == 0:  # 37°C
        Y_LG = p["Y_LG"]
    else:  # 33°C
        Y_LG = p["Y_LG"] * temp_coeffs["Y_LG_factor"]

    return Y_LG * qG


"""def q_ammonium(F_other, Xv, V, p):
    #Удельная скорость образования аммония
    if Xv * V < 1e-6:
        return p["alpha"]
    return p["alpha"] * F_other / (Xv * V)
"""


def q_ammonium(F_other, Xv, V, p, T_phase=0, viability=None):
    """Комбинированная модель образования аммония"""
    if Xv * V < 1e-6:
        return 0.0

    base_rate = p["alpha"] * F_other / (Xv * V)


    #if T_phase == 1:
    #    base_rate = base_rate * 1.1


    if viability is not None and viability < p.get("Vcrit", 0.9):
        viability_factor = 1.0 + (p["Vcrit"] - viability) / p["Vcrit"]
        base_rate = base_rate * viability_factor


    return base_rate

"""def q_product_base(G, viability, p, T_phase, temp_coeffs):
  # Базовая удельная скорость продукции антител
    if T_phase == 0:  # 37°C
        qP_max = p["qP_max"]
    else:  # 33°C
        qP_max = p["qP_max"] * temp_coeffs["qP_max_factor"]

    f_viab = 1 / (1 + np.exp(-p["s"] * (viability - p["Vcrit"])))
    return qP_max * (p["KP"] / (p["KP"] + G)) * f_viab
"""
def q_product_base(G, viability, p, T_phase, temp_coeffs):
    """Базовая удельная скорость продукции антител"""
    if T_phase == 0:  # 37°C
        qP_max = p["qP_max"]
        # До сдвига используем полную формулу
        f_viab = 1 / (1 + np.exp(-p["s"] * (viability - p["Vcrit"])))
        return qP_max * (p["KP"] / (p["KP"] + G)) * f_viab
    else:  # 33°C
        # После сдвига используем УПРОЩЁННУЮ формулу
        # Только от глюкозы, без сильной зависимости от viability
        qP_max = p["qP_max"] * temp_coeffs["qP_max_factor"]
        return qP_max * (p["KP"] / (p["KP"] + G * 0.1))  # Без f_viab!


def q_product_enhanced(G, Lac, NH4_mM, viability, p, T_phase, temp_coeffs):

    # Базовая скорость (как в текущей модели)
    if T_phase == 0:
        qP_base = p["qP_max"] * (p["KP"] / (p["KP"] + G ))
        f_viab = 1 / (1 + np.exp(-p["s"] * (viability - p["Vcrit"])))
    else:
        qP_base = p["qP_max"] * temp_coeffs["qP_max_factor"]
        f_viab = 0.8  # После сдвига зависимость слабее

    # Ингибирование метаболитами (НОВОЕ)
    f_Lac = p["Ki_Lac"] / (p["Ki_Lac"] + Lac) if Lac > 0 else 1.0
    f_NH4 = p["Ki_NH4"] / (p["Ki_NH4"] + NH4_mM) if NH4_mM > 0 else 1.0

    # Комбинированный эффект
    qP = qP_base * f_viab * f_Lac * f_NH4
    #print(qP_base, f_viab, f_Lac, f_NH4)
    return max(qP, 0.0)

def cell_death_rate(viability, p, T_phase, temp_coeffs):
    """Скорость гибели клеток"""
    if T_phase == 0:  # 37°C
        kd0 = p["k_d0"]
    else:  # 33°C
        kd0 = p["k_d0"] * temp_coeffs["k_d0_factor"]

    if viability >= p["Vcrit"]:
        return kd0

    # Проверяем наличие beta в параметрах
    if "beta" in p:
        return kd0 + p["beta"] * (p["Vcrit"] - viability)
    else:
        return kd0 * 1.5  # Упрощённое увеличение при низкой жизнеспособности