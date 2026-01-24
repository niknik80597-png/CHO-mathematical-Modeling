"""
Модуль для визуализации результатов калибровки
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def plot_single_batch_metrics(comparison_df, metrics, batch_id, save_path=None):
    """
    Графики сравнения с метриками на вставках для одной партии
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    variables = [
        ('exp_TCD_1e6_per_mL', 'model_TCD', 'TCD, 10⁶ кл/мл'),
        ('exp_glucose_g_L', 'model_G', 'Глюкоза, г/л'),
        ('exp_lactate_g_L', 'model_Lac', 'Лактат, г/л'),
        ('exp_ammonium_g_L', 'model_NH4_gL', 'Аммоний, г/л'),
        ('exp_titer_g_L', 'model_P', 'Титр, г/л'),
        ('exp_viability_frac', None, 'Жизнеспособность')
    ]

    for idx, (exp_col, model_col, ylabel) in enumerate(variables):
        ax = axes[idx // 3, idx % 3]

        # Экспериментальные данные
        ax.scatter(comparison_df['time_h'], comparison_df[exp_col],
                   color='red', s=50, alpha=0.7, label='Эксперимент', zorder=3)

        # Модельные данные
        if model_col:
            ax.plot(comparison_df['time_h'], comparison_df[model_col],
                    'b-', linewidth=2, label='Модель', zorder=2)
        else:
            # Для жизнеспособности рассчитываем
            model_viab = comparison_df['model_Xv'] / comparison_df['model_TCD']
            ax.plot(comparison_df['time_h'], model_viab,
                    'b-', linewidth=2, label='Модель', zorder=2)

        # Добавляем метрики
        var_name = ['TCD', 'Glucose', 'Lactate', 'Ammonium', 'Titer', 'Viability'][idx]
        if var_name in metrics:
            m = metrics[var_name]
            if not np.isnan(m['R²']):
                text_box = f"R² = {m['R²']:.3f}\nMAE = {m['MAE']:.2f}"
                ax.text(0.05, 0.95, text_box, transform=ax.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Время, ч')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(f'{ylabel.split(",")[0]}', fontsize=11)

    plt.suptitle(f'Калибровка модели: {batch_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_calibration_comparison(all_results, all_metrics, save_path=None):
    """
    Сравнение калибровки между партиями
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. График R² по партиям и параметрам
    ax = axes[0, 0]

    # Подготовка данных
    data = []
    for batch_id, metrics in all_metrics.items():
        for var_name, var_metrics in metrics.items():
            if var_name.startswith('_'):
                continue
            if not np.isnan(var_metrics['R²']):
                data.append({
                    'batch': batch_id,
                    'variable': var_name,
                    'R2': var_metrics['R²']
                })

    if data:
        df = pd.DataFrame(data)

        # Heatmap R²
        pivot = df.pivot(index='variable', columns='batch', values='R2')
        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

        # Добавляем значения
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color='black' if value > 0.7 else 'white', fontsize=9)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title('Коэффициент детерминации R² по партиям', fontsize=12)

        # Цветовая шкала
        plt.colorbar(im, ax=ax, label='R²')

    # 2. Средние метрики по партиям
    ax = axes[0, 1]

    batch_stats = []
    for batch_id, metrics in all_metrics.items():
        if '_summary' in metrics:
            batch_stats.append({
                'batch': batch_id,
                'Mean_R2': metrics['_summary']['Mean_R2'],
                'Mean_MAPE': metrics['_summary']['Mean_MAPE']
            })

    if batch_stats:
        df_stats = pd.DataFrame(batch_stats)
        x = np.arange(len(df_stats))

        ax.bar(x - 0.2, df_stats['Mean_R2'], 0.4, label='Средний R²', alpha=0.8)
        ax.set_ylabel('Средний R²', color='tab:blue')
        ax.set_ylim([0, 1.1])
        ax.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax.twinx()
        ax2.bar(x + 0.2, df_stats['Mean_MAPE'], 0.4, label='Средний MAPE',
                color='tab:orange', alpha=0.8)
        ax2.set_ylabel('Средний MAPE, %', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        ax.set_xticks(x)
        ax.set_xticklabels(df_stats['batch'], rotation=45)
        ax.set_title('Средние метрики по партиям', fontsize=12)

        # Добавляем горизонтальные линии для целевых значений
        ax.axhline(y=0.9, color='tab:blue', linestyle='--', alpha=0.5)
        ax2.axhline(y=15, color='tab:orange', linestyle='--', alpha=0.5)

    # 3. Распределение ошибок
    ax = axes[1, 0]

    all_errors = []
    error_labels = []

    for batch_id, metrics in all_metrics.items():
        for var_name, var_metrics in metrics.items():
            if var_name.startswith('_'):
                continue
            if not np.isnan(var_metrics['MAPE']):
                all_errors.append(var_metrics['MAPE'])
                error_labels.append(f'{batch_id}\n{var_name}')

    if all_errors:
        ax.scatter(range(len(all_errors)), all_errors, alpha=0.7, s=50)
        ax.axhline(y=15, color='r', linestyle='--', label='Допуск 15%')
        ax.axhline(y=np.median(all_errors), color='g', linestyle='-',
                   label=f'Медиана: {np.median(all_errors):.1f}%')

        ax.set_xlabel('Параметр-партия')
        ax.set_ylabel('MAPE, %')
        ax.set_title('Распределение относительных ошибок', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Сравнение конечных титров
    ax = axes[1, 1]

    final_titers = []
    batch_labels = []

    for batch_id, data in all_results.items():
        comparison_df = data['comparison_df']

        # Экспериментальный титр (последний ненулевой)
        exp_titer = comparison_df['exp_titer_g_L'].dropna()
        if len(exp_titer) > 0:
            exp_final = exp_titer.iloc[-1]
            model_final = comparison_df['model_P'].iloc[-1]

            final_titers.append((exp_final, model_final))
            batch_labels.append(batch_id)

    if final_titers:
        exp_finals, model_finals = zip(*final_titers)

        x = np.arange(len(batch_labels))
        width = 0.35

        ax.bar(x - width / 2, exp_finals, width, label='Эксперимент', alpha=0.8)
        ax.bar(x + width / 2, model_finals, width, label='Модель', alpha=0.8)

        # Добавляем процентные различия
        for i, (exp, model) in enumerate(zip(exp_finals, model_finals)):
            diff_pct = abs(exp - model) / exp * 100 if exp > 0 else 0
            ax.text(i, max(exp, model) * 1.05, f'{diff_pct:.1f}%',
                    ha='center', fontsize=8)

        ax.set_xlabel('Партия')
        ax.set_ylabel('Конечный титр, г/л')
        ax.set_title('Сравнение конечных титров', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Сравнительный анализ калибровки модели',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig