
"""
Сравнение механистической и гибридной моделей
"""

import numpy as np
import pandas as pd
import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_model_results(model_type="mech"):
    """
    Загрузка результатов моделирования
    model_type: 'mech' - механистическая, 'hybrid' - гибридная
    """
    if model_type == "mech":
        pattern = "data/processed/simulation_*_comparison.csv"
    else:
        pattern = "data/hybrid_results/simulation_*_hybrid_comparison.csv"

    files = glob.glob(pattern)
    results = {}

    for file_path in files:
        # Извлекаем batch_id
        filename = os.path.basename(file_path)
        if model_type == "mech":
            batch_id = filename.replace('simulation_', '').replace('_comparison.csv', '')
        else:
            batch_id = filename.replace('simulation_', '').replace('_hybrid_comparison.csv', '')

        try:
            df = pd.read_csv(file_path)
            results[batch_id] = df
            print(f"  ✅ {batch_id}: {len(df)} точек")
        except Exception as e:
            print(f"  ❌ {batch_id}: ошибка загрузки - {str(e)}")

    return results


def calculate_comparison_metrics(mech_results, hybrid_results):
    """
    Вычисление метрик сравнения
    """
    comparison_metrics = {}
    batch_comparisons = {}

    # Общие партии
    common_batches = set(mech_results.keys()) & set(hybrid_results.keys())

    if not common_batches:
        print("❌ Нет общих партий для сравнения!")
        return None

    print(f"\n📊 СРАВНЕНИЕ ПО {len(common_batches)} ПАРТИЯМ:")

    for batch_id in common_batches:
        mech_df = mech_results[batch_id]
        hybrid_df = hybrid_results[batch_id]

        # Выравниваем временные точки
        common_times = set(mech_df['time_h']) & set(hybrid_df['time_h'])

        if not common_times:
            print(f"  ⚠️  {batch_id}: нет общих временных точек")
            continue

        # Вычисляем метрики для титра
        mech_titer = mech_df.set_index('time_h')['model_P']
        hybrid_titer = hybrid_df.set_index('time_h')['model_P']
        exp_titer = mech_df.set_index('time_h')['exp_titer_g_L'].dropna()

        # Общие временные точки с экспериментальными данными
        valid_times = exp_titer.index.intersection(mech_titer.index).intersection(hybrid_titer.index)

        if len(valid_times) < 2:
            print(f"  ⚠️  {batch_id}: недостаточно данных для сравнения")
            continue

        # Вычисляем R²
        from sklearn.metrics import r2_score, mean_absolute_percentage_error

        mech_r2 = r2_score(exp_titer.loc[valid_times], mech_titer.loc[valid_times])
        hybrid_r2 = r2_score(exp_titer.loc[valid_times], hybrid_titer.loc[valid_times])

        # Вычисляем MAPE
        mech_mape = mean_absolute_percentage_error(exp_titer.loc[valid_times], mech_titer.loc[valid_times]) * 100
        hybrid_mape = mean_absolute_percentage_error(exp_titer.loc[valid_times], hybrid_titer.loc[valid_times]) * 100

        # Сохраняем метрики для партии
        batch_comparisons[batch_id] = {
            'mech_r2': mech_r2,
            'hybrid_r2': hybrid_r2,
            'mech_mape': mech_mape,
            'hybrid_mape': hybrid_mape,
            'r2_improvement': hybrid_r2 - mech_r2,
            'mape_improvement': mech_mape - hybrid_mape,  # Уменьшение ошибки
            'n_points': len(valid_times),
            'improvement_percent': ((hybrid_r2 - mech_r2) / abs(mech_r2) * 100) if mech_r2 != 0 else 0
        }

        print(f"  📈 {batch_id}:")
        print(f"     Механистическая: R²={mech_r2:.3f}, MAPE={mech_mape:.1f}%")
        print(f"     Гибридная:       R²={hybrid_r2:.3f}, MAPE={hybrid_mape:.1f}%")
        print(
            f"     Улучшение:       ΔR²={hybrid_r2 - mech_r2:.3f} ({batch_comparisons[batch_id]['improvement_percent']:.1f}%)")

    # Сводные метрики
    if batch_comparisons:
        comparison_metrics = {
            'batch_comparisons': batch_comparisons,
            'mech_mean_r2': np.mean([v['mech_r2'] for v in batch_comparisons.values()]),
            'hybrid_mean_r2': np.mean([v['hybrid_r2'] for v in batch_comparisons.values()]),
            'mech_mean_mape': np.mean([v['mech_mape'] for v in batch_comparisons.values()]),
            'hybrid_mean_mape': np.mean([v['hybrid_mape'] for v in batch_comparisons.values()]),
            'r2_improvement': np.mean([v['hybrid_r2'] - v['mech_r2'] for v in batch_comparisons.values()]),
            'mape_improvement': np.mean([v['mech_mape'] - v['hybrid_mape'] for v in batch_comparisons.values()]),
            'n_batches': len(batch_comparisons),
            'comparison_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n🎯 СВОДНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"  Средний R² (механистическая): {comparison_metrics['mech_mean_r2']:.3f}")
        print(f"  Средний R² (гибридная):       {comparison_metrics['hybrid_mean_r2']:.3f}")
        print(f"  Улучшение R²:                 {comparison_metrics['r2_improvement']:.3f}")
        print(f"  Улучшение MAPE:               {comparison_metrics['mape_improvement']:.1f}%")

    return comparison_metrics


def create_comparison_plots(comparison_metrics):
    """
    Создание графиков сравнения
    """
    output_dir = "data/comparison/plots"
    os.makedirs(output_dir, exist_ok=True)

    batch_comparisons = comparison_metrics.get('batch_comparisons', {})

    if not batch_comparisons:
        print("❌ Нет данных для графиков!")
        return

    # 1. График сравнения R² по партиям
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    batches = list(batch_comparisons.keys())
    mech_r2 = [batch_comparisons[b]['mech_r2'] for b in batches]
    hybrid_r2 = [batch_comparisons[b]['hybrid_r2'] for b in batches]
    improvements = [batch_comparisons[b]['r2_improvement'] for b in batches]

    # График 1: R² по партиям
    ax = axes[0, 0]
    x = np.arange(len(batches))
    width = 0.35

    ax.bar(x - width / 2, mech_r2, width, label='Механистическая', alpha=0.8, color='steelblue')
    ax.bar(x + width / 2, hybrid_r2, width, label='Гибридная', alpha=0.8, color='coral')

    ax.set_xlabel('Партия')
    ax.set_ylabel('R²')
    ax.set_title('Сравнение R² по партиям')
    ax.set_xticks(x)
    ax.set_xticklabels(batches, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Добавляем линии улучшения
    for i, (m, h) in enumerate(zip(mech_r2, hybrid_r2)):
        if h > m:
            ax.plot([i - width / 2, i + width / 2], [m, h], 'g-', linewidth=2, alpha=0.7)

    # График 2: Улучшение R²
    ax = axes[0, 1]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(batches, improvements, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Партия')
    ax.set_ylabel('Улучшение R² (гибридная - механистическая)')
    ax.set_title('Улучшение качества предсказания')
    ax.set_xticklabels(batches, rotation=45)
    ax.grid(True, alpha=0.3)

    # Добавляем значения на бары
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + (0.01 if height >= 0 else -0.02),
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    # График 3: Сводное сравнение
    ax = axes[1, 0]
    categories = ['R² (мех)', 'R² (гибр)', 'MAPE (мех)', 'MAPE (гибр)']
    values = [
        comparison_metrics['mech_mean_r2'],
        comparison_metrics['hybrid_mean_r2'],
        comparison_metrics['mech_mean_mape'],
        comparison_metrics['hybrid_mean_mape']
    ]

    colors = ['steelblue', 'coral', 'steelblue', 'coral']
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Значение')
    ax.set_title('Средние метрики по всем партиям')
    ax.grid(True, alpha=0.3)

    # Добавляем значения на бары
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{val:.3f}' if 'R²' in bar.get_label() else f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)

    # График 4: Корреляция улучшений
    ax = axes[1, 1]

    if len(batches) >= 3:
        # Рассчитываем улучшение для каждой партии
        improvements_array = np.array(improvements)

        # Если есть улучшения
        if np.any(improvements_array > 0):
            ax.scatter(mech_r2, hybrid_r2, s=100, alpha=0.7, edgecolor='black')

            # Добавляем подписи партий
            for i, batch in enumerate(batches):
                ax.annotate(batch, (mech_r2[i], hybrid_r2[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)

            # Линия идеального равенства
            min_r2 = min(min(mech_r2), min(hybrid_r2))
            max_r2 = max(max(mech_r2), max(hybrid_r2))
            ax.plot([min_r2, max_r2], [min_r2, max_r2], 'r--', alpha=0.5, label='Равенство')

            ax.set_xlabel('R² механистической модели')
            ax.set_ylabel('R² гибридной модели')
            ax.set_title('Корреляция качества моделей')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Нет улучшений\nдля визуализации',
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, 'Недостаточно данных\nдля корреляции',
                ha='center', va='center', fontsize=12)
        ax.set_axis_off()

    plt.suptitle('Сравнение механистической и гибридной моделей',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Детальные графики по партиям
    create_detailed_batch_plots(batch_comparisons, output_dir)

    print(f"\n📊 Графики сохранены в: {output_dir}/")


def create_detailed_batch_plots(batch_comparisons, output_dir):
    """
    Создание детальных графиков по каждой партии
    """
    detailed_dir = os.path.join(output_dir, "batch_details")
    os.makedirs(detailed_dir, exist_ok=True)

    for batch_id, metrics in batch_comparisons.items():
        # Загружаем данные обеих моделей
        mech_file = f"data/processed/simulation_{batch_id}_comparison.csv"
        hybrid_file = f"data/hybrid_results/simulation_{batch_id}_hybrid_comparison.csv"

        if not (os.path.exists(mech_file) and os.path.exists(hybrid_file)):
            continue

        mech_df = pd.read_csv(mech_file)
        hybrid_df = pd.read_csv(hybrid_file)

        # Создаем график
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # График 1: Сравнение титра
        ax = axes[0, 0]

        # Экспериментальные данные
        exp_times = mech_df['time_h'][~mech_df['exp_titer_g_L'].isna()]
        exp_titer = mech_df['exp_titer_g_L'][~mech_df['exp_titer_g_L'].isna()]
        ax.scatter(exp_times, exp_titer, color='black', s=50, alpha=0.7,
                   label='Эксперимент', zorder=3)

        # Модельные данные
        ax.plot(mech_df['time_h'], mech_df['model_P'], 'b-', linewidth=2,
                label='Механистическая', alpha=0.8, zorder=2)
        ax.plot(hybrid_df['time_h'], hybrid_df['model_P'], 'r-', linewidth=2,
                label='Гибридная', alpha=0.8, zorder=2)

        ax.set_xlabel('Время, ч')
        ax.set_ylabel('Титр, г/л')
        ax.set_title(f'Сравнение титра: {batch_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Добавляем метрики
        text_box = f"R² мех: {metrics['mech_r2']:.3f}\nR² гибр: {metrics['hybrid_r2']:.3f}\nУлучшение: {metrics['r2_improvement']:.3f}"
        ax.text(0.05, 0.95, text_box, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # График 2: ML коррекция (если есть)
        ax = axes[0, 1]

        if 'ml_correction' in hybrid_df.columns:
            ax.plot(hybrid_df['time_h'], hybrid_df['ml_correction'] * 100,
                    'g-', linewidth=2, alpha=0.8)
            ax.fill_between(hybrid_df['time_h'], 0, hybrid_df['ml_correction'] * 100,
                            alpha=0.3, color='green')

            ax.set_xlabel('Время, ч')
            ax.set_ylabel('ML коррекция, %')
            ax.set_title('ML коррекция титра')
            ax.grid(True, alpha=0.3)

            # Средняя коррекция
            mean_correction = hybrid_df['ml_correction'].mean() * 100
            ax.axhline(y=mean_correction, color='red', linestyle='--', alpha=0.7,
                       label=f'Средняя: {mean_correction:.1f}%')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Нет данных о ML коррекции',
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()

        # График 3: Ошибки предсказания
        ax = axes[1, 0]

        # Вычисляем ошибки
        valid_times = exp_times[exp_times.isin(mech_df['time_h']) & exp_times.isin(hybrid_df['time_h'])]

        if len(valid_times) > 0:
            mech_errors = []
            hybrid_errors = []

            for t in valid_times:
                exp_val = exp_titer[exp_times == t].iloc[0]
                mech_val = mech_df.loc[mech_df['time_h'] == t, 'model_P'].iloc[0]
                hybrid_val = hybrid_df.loc[hybrid_df['time_h'] == t, 'model_P'].iloc[0]

                mech_errors.append(abs(exp_val - mech_val))
                hybrid_errors.append(abs(exp_val - hybrid_val))

            x = np.arange(len(valid_times))
            width = 0.35

            ax.bar(x - width / 2, mech_errors, width, label='Механистическая',
                   alpha=0.7, color='steelblue')
            ax.bar(x + width / 2, hybrid_errors, width, label='Гибридная',
                   alpha=0.7, color='coral')

            ax.set_xlabel('Порядковый номер измерения')
            ax.set_ylabel('Абсолютная ошибка, г/л')
            ax.set_title('Ошибки предсказания титра')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Уменьшаем количество меток на оси X
            if len(valid_times) > 10:
                ax.set_xticks(x[::len(x) // 10])
                ax.set_xticklabels([f'{int(t)}' for t in valid_times.iloc[::len(valid_times) // 10]])
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([f'{int(t)}' for t in valid_times])
        else:
            ax.text(0.5, 0.5, 'Нет данных для сравнения ошибок',
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()

        # График 4: Кумулятивное улучшение
        ax = axes[1, 1]

        if 'ml_correction_abs' in hybrid_df.columns:
            cumulative_improvement = hybrid_df['ml_correction_abs'].cumsum()

            ax.plot(hybrid_df['time_h'], cumulative_improvement,
                    'purple', linewidth=2, alpha=0.8)
            ax.fill_between(hybrid_df['time_h'], 0, cumulative_improvement,
                            alpha=0.3, color='purple')

            ax.set_xlabel('Время, ч')
            ax.set_ylabel('Кумулятивное улучшение титра, г/л')
            ax.set_title('Накопленный эффект ML коррекции')
            ax.grid(True, alpha=0.3)

            # Финальное улучшение
            final_improvement = cumulative_improvement.iloc[-1]
            ax.text(0.05, 0.95, f'Финальное улучшение: {final_improvement:.2f} г/л',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Нет данных о кумулятивном улучшении',
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()

        plt.suptitle(f'Детальное сравнение моделей: {batch_id}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Сохраняем график
        plt.savefig(f"{detailed_dir}/{batch_id}_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)  # Закрываем график для экономии памяти


def compare_model_performance():
    """
    Основная функция сравнения моделей
    """
    print(f"{'=' * 80}")
    print("📊 СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ")
    print(f"{'=' * 80}")

    # Загружаем результаты
    print("\n🔍 Загрузка результатов механистической модели...")
    mech_results = load_model_results("mech")

    print("\n🔍 Загрузка результатов гибридной модели...")
    hybrid_results = load_model_results("hybrid")

    if not mech_results or not hybrid_results:
        print("❌ Не удалось загрузить результаты одной из моделей!")
        return None

    # Вычисляем метрики сравнения
    comparison_metrics = calculate_comparison_metrics(mech_results, hybrid_results)

    if comparison_metrics:
        # Сохраняем результаты
        output_dir = "data/comparison"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/comparison_results.json", "w") as f:
            json.dump(comparison_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Результаты сравнения сохранены: {output_dir}/comparison_results.json")

    return comparison_metrics


if __name__ == "__main__":
    results = compare_model_performance()

    if results:
        create_comparison_plots(results)
        print("\n🎉 Сравнение моделей завершено успешно!")
    else:
        print("\n❌ Сравнение моделей не удалось")
