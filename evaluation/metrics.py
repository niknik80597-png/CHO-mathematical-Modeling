"""
Модуль для расчета метрик калибровки - ИСПРАВЛЕННАЯ ВЕРСИЯ (БЕЗ ГЛЮКОЗЫ)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_calibration_metrics(comparison_df, batch_id, exclude_glucose=True):
    """
    Расчет статистических метрик калибровки - ИСПРАВЛЕННАЯ ВЕРСИЯ
    С опцией исключения глюкозы из общей статистики

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame с колонками exp_* и model_*
    batch_id : str
        Идентификатор партии
    exclude_glucose : bool
        Исключать ли глюкозу из расчета сводных метрик (True по умолчанию)

    Returns:
    --------
    dict: Словарь с метриками для каждого параметра
        Структура: {variable_name: {'MAE': ..., 'RMSE': ..., 'R²': ..., 'MAPE': ...}}
    """

    metrics = {}

    # Определяем переменные для оценки
    variables = {
        'TCD': ('exp_TCD_1e6_per_mL', 'model_TCD'),
        'Glucose': ('exp_glucose_g_L', 'model_G'),
        'Lactate': ('exp_lactate_g_L', 'model_Lac'),
        'Ammonium': ('exp_ammonium_g_L', 'model_NH4_gL'),
        'Titer': ('exp_titer_g_L', 'model_P'),
        'Viability': ('exp_viability_frac', None)
    }

    for var_name, (exp_col, model_col) in variables.items():
        try:
            if var_name == 'Viability':
                # Для жизнеспособности рассчитываем из модели
                model_viab = comparison_df['model_Xv'] / comparison_df['model_TCD']
                exp_viab = comparison_df[exp_col]

                # Фильтруем NaN
                mask = ~np.isnan(exp_viab) & ~np.isnan(model_viab)
                y_true = exp_viab[mask]
                y_pred = model_viab[mask]
            else:
                # Для остальных переменных
                exp_data = comparison_df[exp_col]
                model_data = comparison_df[model_col]

                # Фильтруем NaN
                mask = ~np.isnan(exp_data)
                y_true = exp_data[mask]
                y_pred = model_data[mask]

            # Проверяем, что есть достаточно данных для сравнения
            if len(y_true) >= 2 and len(y_pred) >= 2 and len(y_true) == len(y_pred):
                # Основные метрики
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)

                # R² (с проверкой)
                try:
                    r2 = r2_score(y_true, y_pred)
                except Exception:
                    r2 = np.nan

                # MAPE (средняя абсолютная процентная ошибка)
                # Избегаем деления на 0
                mask_nonzero = y_true != 0
                if np.sum(mask_nonzero) > 0:
                    mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) /
                                         y_true[mask_nonzero])) * 100
                else:
                    mape = np.nan

                metrics[var_name] = {
                    'MAE': float(mae),
                    'MSE': float(mse),
                    'RMSE': float(rmse),
                    'R²': float(r2) if not np.isnan(r2) else np.nan,
                    'MAPE': float(mape) if not np.isnan(mape) else np.nan,
                    'N_points': int(len(y_true)),
                    'Min_true': float(y_true.min()) if len(y_true) > 0 else np.nan,
                    'Max_true': float(y_true.max()) if len(y_true) > 0 else np.nan,
                    'Min_pred': float(y_pred.min()) if len(y_pred) > 0 else np.nan,
                    'Max_pred': float(y_pred.max()) if len(y_pred) > 0 else np.nan
                }
            else:
                # Недостаточно данных
                metrics[var_name] = {
                    'MAE': np.nan,
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R²': np.nan,
                    'MAPE': np.nan,
                    'N_points': 0,
                    'Min_true': np.nan,
                    'Max_true': np.nan,
                    'Min_pred': np.nan,
                    'Max_pred': np.nan
                }
        except Exception as e:
            # В случае любой ошибки возвращаем NaN
            print(f"⚠️  Ошибка расчета метрик для {var_name} в {batch_id}: {str(e)}")
            metrics[var_name] = {
                'MAE': np.nan,
                'MSE': np.nan,
                'RMSE': np.nan,
                'R²': np.nan,
                'MAPE': np.nan,
                'N_points': 0,
                'Min_true': np.nan,
                'Max_true': np.nan,
                'Min_pred': np.nan,
                'Max_pred': np.nan
            }

    # Дополнительные сводные метрики (БЕЗ ГЛЮКОЗЫ)
    r2_values = []
    mape_values = []

    for var_name, var_metrics in metrics.items():
        # Исключаем глюкозу из сводной статистики
        if exclude_glucose and var_name == 'Glucose':
            continue

        if var_name.startswith('_'):
            continue

        if not np.isnan(var_metrics['R²']):
            r2_values.append(var_metrics['R²'])
        if not np.isnan(var_metrics['MAPE']):
            mape_values.append(var_metrics['MAPE'])

    if r2_values:
        metrics['_summary'] = {
            'Mean_R2': float(np.nanmean(r2_values)),
            'Std_R2': float(np.nanstd(r2_values)),
            'Mean_MAPE': float(np.nanmean(mape_values)) if mape_values else np.nan,
            'Std_MAPE': float(np.nanstd(mape_values)) if mape_values else np.nan,
            'N_variables': len([m for m in metrics.values() if isinstance(m, dict) and m.get('N_points', 0) > 0]),
            'Glucose_excluded': exclude_glucose
        }
    else:
        metrics['_summary'] = {
            'Mean_R2': np.nan,
            'Std_R2': np.nan,
            'Mean_MAPE': np.nan,
            'Std_MAPE': np.nan,
            'N_variables': 0,
            'Glucose_excluded': exclude_glucose
        }

    return metrics

def create_summary_table(all_metrics):
    """
    Создание сводной таблицы метрик для всех партий - ИСПРАВЛЕННАЯ ВЕРСИЯ

    Parameters:
    -----------
    all_metrics : dict
        Словарь {batch_id: metrics} для всех партий

    Returns:
    --------
    pd.DataFrame: Сводная таблица
    """

    summary_data = []

    for batch_id, metrics in all_metrics.items():
        for var_name, var_metrics in metrics.items():
            # Пропускаем summary
            if var_name.startswith('_'):
                continue

            # Проверяем, что var_metrics - это словарь и содержит нужные ключи
            if not isinstance(var_metrics, dict):
                print(f"⚠️  Пропускаем {var_name} в {batch_id}: не словарь")
                continue

            # Проверяем наличие ключей
            required_keys = ['MAE', 'RMSE', 'R²', 'MAPE']
            missing_keys = [key for key in required_keys if key not in var_metrics]

            if missing_keys:
                print(f"⚠️  Пропускаем {var_name} в {batch_id}: отсутствуют ключи {missing_keys}")
                continue

            # Получаем значения, обрабатывая NaN
            mae = var_metrics['MAE']
            rmse = var_metrics['RMSE']
            r2 = var_metrics['R²']
            mape = var_metrics['MAPE']

            summary_data.append({
                'Партия': batch_id,
                'Параметр': var_name,
                'MAE': f"{mae:.3f}" if not np.isnan(mae) else "NaN",
                'RMSE': f"{rmse:.3f}" if not np.isnan(rmse) else "NaN",
                'R²': f"{r2:.3f}" if not np.isnan(r2) else "NaN",
                'MAPE, %': f"{mape:.1f}" if not np.isnan(mape) else "NaN",
                'Точек': var_metrics.get('N_points', 0)
            })

    return pd.DataFrame(summary_data)

def calculate_overall_metrics(all_metrics, exclude_glucose=True):
    """
    Расчет общих метрик по всем партиям - ИСПРАВЛЕННАЯ ВЕРСИЯ
    С опцией исключения глюкозы

    Parameters:
    -----------
    all_metrics : dict
        Словарь метрик для всех партий
    exclude_glucose : bool
        Исключать ли глюкозу из расчета общих метрик

    Returns:
    --------
    dict: Общие метрики
    """

    all_r2 = []
    all_mape = []

    for batch_id, batch_metrics in all_metrics.items():
        for var_name, var_metrics in batch_metrics.items():
            if var_name.startswith('_'):
                continue

            if not isinstance(var_metrics, dict):
                continue

            # Исключаем глюкозу из общих метрик
            if exclude_glucose and var_name == 'Glucose':
                continue

            r2 = var_metrics.get('R²', np.nan)
            mape = var_metrics.get('MAPE', np.nan)

            if not np.isnan(r2):
                all_r2.append(r2)

            if not np.isnan(mape):
                all_mape.append(mape)

    if all_r2 and all_mape:
        return {
            'Overall_R2_mean': float(np.mean(all_r2)),
            'Overall_R2_std': float(np.std(all_r2)),
            'Overall_MAPE_mean': float(np.mean(all_mape)),
            'Overall_MAPE_std': float(np.std(all_mape)),
            'N_batches': len(all_metrics),
            'N_datapoints': len(all_r2),
            'Glucose_excluded': exclude_glucose
        }
    else:
        return {
            'Overall_R2_mean': np.nan,
            'Overall_R2_std': np.nan,
            'Overall_MAPE_mean': np.nan,
            'Overall_MAPE_std': np.nan,
            'N_batches': len(all_metrics),
            'N_datapoints': 0,
            'Glucose_excluded': exclude_glucose
        }