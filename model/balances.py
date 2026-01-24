import numpy as np
"""
def volume_balance(V, F_glc, F_other, dt, rho):
    #Баланс объёма биореактора
    dV = (F_glc + F_other) /1000 * dt / rho/1000
    return V + dV
"""

def volume_balance(V, F_glc, F_other, dt, rho):
    #Баланс объёма биореактора
    # F_glc и F_other в г/ч
    # Предполагаем, что это растворы:
    # Глюкоза: 500 г/л раствор
    # Другие: 200 г/л раствор
    F_glc_Lph = F_glc / 500.0  # г/ч → л/ч
    F_other_Lph = F_other / 200.0  # г/ч → л/ч

    dV = (F_glc_Lph + F_other_Lph) * dt
    return V + dV


def glucose_balance(G, qG, Xv, V, F_glc, dVdt, dt):
    """Баланс глюкозы"""
    # Приток - потребление - разбавление
    dG = (F_glc / V - qG * Xv - G * dVdt / V) * dt
    return max(G + dG, 0.0)

def lactate_balance(Lac, qL, Xv, V, dVdt, dt):
    """Баланс лактата"""
    # Образование - разбавление
    dLac = (qL * Xv - Lac * dVdt / V) * dt
    return max(Lac + dLac, 0.0)

def ammonium_balance(NH4, qNH4, Xv, V, dVdt, dt):
    """Баланс аммония"""
    # Образование - разбавление
    dNH4 = (qNH4 * Xv - NH4 * dVdt / V) * dt
    return max(NH4 + dNH4, 0.0)

def product_balance(P, qP, Xv, V, dVdt, dt):
    """Баланс продукта (мАт)"""
    # Продукция - разбавление
    dP = (qP * Xv - P * dVdt / V) * dt
    return max(P + dP, 0.0)

def biomass_balance(Xv, mu, kd, V, dVdt, dt):
    """Баланс жизнеспособной биомассы"""
    # Рост - гибель - разбавление
    # Исправлено: убрано умножение Xv на dVdt/V внутри второго члена
    dXv = (mu - kd) * Xv * dt - (dVdt / V) * Xv * dt
    return max(Xv + dXv, 0.0)

def total_cell_density_balance(TCD, mu, kd, V, dVdt, dt):
    """Баланс общей плотности клеток"""
    # TCD включает все клетки (живые + мёртвые)
    # Гибель должна вычитать клетки из TCD
    dTCD = mu * TCD * dt - (dVdt / V) * TCD * dt
    # kd учитывается только в Xv, не в TCD
    return max(TCD + dTCD, 0.0)