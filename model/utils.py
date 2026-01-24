import numpy as np
from scipy.interpolate import interp1d


class DataInterpolator:
    """Класс для интерполяции экспериментальных данных"""

    def __init__(self, time_points, data_points, kind='linear'):
        self.interpolator = interp1d(
            time_points,
            data_points,
            kind=kind,
            bounds_error=False,
            fill_value=(data_points[0], data_points[-1])
        )

    def __call__(self, t):
        return float(self.interpolator(t))


def create_interpolators_from_df(df, columns):
    """Создание интерполяторов для указанных колонок DataFrame"""
    interpolators = {}
    time_points = df['time_h'].values

    for col in columns:
        if col in df.columns:
            data_points = df[col].values
            interpolators[col] = DataInterpolator(time_points, data_points)

    return interpolators


def calculate_dVdt(V, V_prev, dt):
    """Расчёт скорости изменения объёма"""
    return (V - V_prev) / dt if dt > 0 else 0.0