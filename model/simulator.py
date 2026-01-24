import numpy as np
from model.balances import *
from model.kinetics import *


def temperature_phase(t, t_shift):
    """Определение температурной фазы"""
    return 0 if t < t_shift else 1


def simulation_step(state, inputs, params, dt, t_shift, temp_coeffs):
    """
    Один шаг моделирования fed-batch процесса

    Parameters:
    -----------
    state : dict
        Текущее состояние системы
    inputs : tuple
        (F_glc, F_other) - скорости подачи
    params : dict
        Параметры модели
    dt : float
        Шаг времени
    t_shift : float
        Время температурного шифта
    temp_coeffs : dict
        Температурные коэффициенты

    Returns:
    --------
    new_state : dict
        Новое состояние системы
    rates : dict
        Рассчитанные скорости процессов
    """

    # Распаковка состояния
    t = state["time_h"]
    V = state["V"]
    Xv = state["Xv"]
    TCD = state["TCD"]
    G = state["G"]
    Lac = state["Lac"]
    NH4 = state["NH4"]
    P = state["P"]

    # Распаковка входных данных
    F_glc, F_other = inputs

    if state["G"] > 2.5:  # Если глюкоза выше 2.5 г/л
        F_glc = 0.0  # Прекращаем подачу
    if t >= t_shift:
        F_glc = F_glc * 0.7  # Уменьшаем подачу после сдвига

    # Определение температурной фазы
    T_phase = temperature_phase(t, t_shift)

    mu = mu_growth(G, Lac, NH4, params, T_phase, temp_coeffs)
    qG = q_glucose(mu, params, T_phase, temp_coeffs)
    qL = q_lactate(qG, params, T_phase, temp_coeffs)
    qNH4 = q_ammonium(F_other, Xv, V, params, T_phase, Xv/TCD)
    viability = compute_viability(Xv, TCD)
    #qP = q_product_base(G, viability, params, T_phase, temp_coeffs)
    qP = q_product_enhanced(G, Lac, NH4 / 0.018, viability, params, T_phase, temp_coeffs)
    kd = cell_death_rate(viability, params, T_phase, temp_coeffs)

    # РАСЧЁТ ИЗМЕНЕНИЯ ОБЪЁМА (сначала!)
    V_new = volume_balance(V, F_glc, F_other, dt, params["rho"])
    dVdt = (V_new - V) / dt if dt > 0 else 0.0

    # РАСЧЁТ НОВЫХ СОСТОЯНИЙ (в правильном порядке!)
    # Сначала TCD (общая плотность клеток)
    TCD_new = total_cell_density_balance(TCD, mu, kd, V, dVdt, dt)

    # Затем Xv (жизнеспособные клетки) - зависит от TCD_new
    Xv_new = biomass_balance(Xv, mu, kd, V, dVdt, dt)

    # Ограничиваем Xv, чтобы не превышал TCD
    Xv_new = min(Xv_new, TCD_new)

    # Затем метаболиты
    G_new = glucose_balance(G, qG, Xv, V, F_glc, dVdt, dt)
    Lac_new = lactate_balance(Lac, qL, Xv, V, dVdt, dt)
    NH4_new = ammonium_balance(NH4, qNH4, Xv, V, dVdt, dt)
    P_new = product_balance(P, qP, Xv, V, dVdt, dt)

    # Пересчитываем жизнеспособность
    viability_new = compute_viability(Xv_new, TCD_new) if TCD_new > 0 else 0.0

    # Новое состояние
    new_state = {
        "time_h": t + dt,
        "V": V_new,
        "Xv": Xv_new,
        "TCD": TCD_new,
        "G": G_new,
        "Lac": Lac_new,
        "NH4": NH4_new,
        "P": P_new,
        "viability": compute_viability(Xv_new, TCD_new)
    }

    # Рассчитанные скорости (для анализа)
    rates = {
        "mu": mu,
        "qG": qG,
        "qL": qL,
        "qNH4": qNH4,
        "qP": qP,
        "kd": kd,
        "dVdt": dVdt,
        "T_phase": T_phase
    }

    return new_state, rates