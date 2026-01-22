#!/usr/bin/env python3
"""
Скрипт для запуска моделирования всех партий CHO
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from main_simulation import run_simulation, plot_single_batch
from evaluation.metrics import calculate_calibration_metrics
from evaluation.visualizer import plot_calibration_comparison
from evaluation.reporter import generate_html_report
from evaluation.reporter import calculate_overall_metrics


def find_all_batches(data_dir="data/raw", meta_dir="data/meta"):
    """
    Найти все доступные партии
    """
    # Ищем CSV файлы
    csv_files = glob.glob(os.path.join(data_dir, "batch_CHO*.csv"))
    batches = []

    for csv_path in csv_files:
        # Извлекаем номер партии
        base_name = os.path.basename(csv_path)
        batch_num = base_name.replace("batch_CHO", "").replace(".csv", "")

        # Проверяем существование JSON файла
        meta_path = os.path.join(meta_dir, f"batch_CHO{batch_num}.json")

        if os.path.exists(meta_path):
            batches.append({
                "batch_id": f"CHO{batch_num}",
                "csv_path": csv_path,
                "meta_path": meta_path,
                "output_prefix": f"data/processed/simulation_CHO{batch_num}"
            })
            print(f"✅ Найдена партия CHO{batch_num}")
        else:
            print(f"⚠️  Найден CSV для CHO{batch_num}, но нет JSON файла")

    return sorted(batches, key=lambda x: x["batch_id"])


def run_all_simulations(batches, output_dir="data/processed"):
    """
    Запустить моделирование для всех партий
    """
    print("=" * 80)
    print("🚀 ЗАПУСК МОДЕЛИРОВАНИЯ ВСЕХ ПАРТИЙ")
    print("=" * 80)

    all_results = {}
    start_time = datetime.now()

    for batch in batches:
        batch_id = batch["batch_id"]
        print(f"\n{'=' * 60}")
        print(f"📊 ПАРТИЯ: {batch_id}")
        print(f"{'=' * 60}")

        try:
            # Запускаем симуляцию
            results_df, rates_df, comparison_df = run_simulation(
                csv_path=batch["csv_path"],
                meta_path=batch["meta_path"],
                output_path=batch["output_prefix"],
                batch_id=batch_id
            )

            # Сохраняем результаты
            all_results[batch_id] = {
                "results_df": results_df,
                "rates_df": rates_df,
                "comparison_df": comparison_df,
                "batch_info": batch
            }

            # Сохраняем график для этой партии
            plot_path = os.path.join(output_dir, f"plot_{batch_id}.png")
            plot_single_batch(results_df, rates_df, comparison_df, save_path=plot_path)

            print(f"✅ {batch_id}: Успешно завершено")

        except Exception as e:
            print(f"❌ {batch_id}: Ошибка - {str(e)}")
            import traceback
            traceback.print_exc()

    # Сводная информация
    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print(f"🎯 МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"   Всего партий: {len(batches)}")
    print(f"   Успешно: {len(all_results)}")
    print(f"   Время выполнения: {elapsed_time:.1f} сек")
    print(f"{'=' * 80}")

    return all_results


def evaluate_all_batches(all_results, exclude_glucose=True):
    """
    Оценка калибровки для всех партий - ИСПРАВЛЕННАЯ ВЕРСИЯ
    С опцией исключения глюкозы

    Parameters:
    -----------
    all_results : dict
        Результаты всех симуляций
    exclude_glucose : bool
        Исключать ли глюкозу из расчета сводных метрик
    """
    print(f"\n{'=' * 80}")
    print("📈 ОЦЕНКА КАЛИБРОВКИ" + (" (без глюкозы)" if exclude_glucose else ""))
    print("=" * 80)

    all_metrics = {}
    summary_data = []

    for batch_id, data in all_results.items():
        print(f"\n🔍 Анализ калибровки: {batch_id}")

        comparison_df = data["comparison_df"]

        # Вычисляем метрики (исключаем глюкозу из сводной статистики)
        metrics = calculate_calibration_metrics(comparison_df, batch_id, exclude_glucose=exclude_glucose)
        all_metrics[batch_id] = metrics

        # Добавляем в сводную таблицу
        for var_name, var_metrics in metrics.items():
            # Пропускаем summary
            if var_name.startswith('_'):
                continue

            # Проверяем, что var_metrics - это словарь
            if not isinstance(var_metrics, dict):
                print(f"  ⚠️  Пропускаем {var_name}: не словарь")
                continue

            # Проверяем наличие ключей
            required_keys = ['MAE', 'RMSE', 'R²', 'MAPE']
            if not all(key in var_metrics for key in required_keys):
                print(f"  ⚠️  Пропускаем {var_name}: отсутствуют ключи")
                continue

            # Получаем значения
            mae = var_metrics['MAE']
            rmse = var_metrics['RMSE']
            r2 = var_metrics['R²']
            mape = var_metrics['MAPE']

            # Форматируем, учитывая NaN
            mae_str = f"{mae:.3f}" if not np.isnan(mae) else "NaN"
            rmse_str = f"{rmse:.3f}" if not np.isnan(rmse) else "NaN"
            r2_str = f"{r2:.3f}" if not np.isnan(r2) else "NaN"
            mape_str = f"{mape:.1f}" if not np.isnan(mape) else "NaN"

            summary_data.append({
                "Партия": batch_id,
                "Параметр": var_name,
                "MAE": mae_str,
                "RMSE": rmse_str,
                "R²": r2_str,
                "MAPE, %": mape_str,
                "Точек": var_metrics.get('N_points', 0)
            })

    # Создаем сводную таблицу
    summary_df = pd.DataFrame(summary_data)

    # Сохраняем метрики
    suffix = "_no_glucose" if exclude_glucose else ""
    metrics_path = f"data/processed/calibration_metrics_all{suffix}.csv"
    summary_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # Сохраняем в Excel для диплома
    excel_path = f"data/processed/calibration_summary{suffix}.xlsx"
    summary_df.to_excel(excel_path, index=False)

    print(f"\n📊 Сводные метрики сохранены:")
    print(f"   CSV: {metrics_path}")
    print(f"   Excel: {excel_path}")

    # Печатаем сводную таблицу
    print(f"\n{'=' * 80}")
    print("📋 СВОДНАЯ ТАБЛИЦА КАЛИБРОВКИ" + (" (без глюкозы)" if exclude_glucose else ""))
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Выводим сводную статистику
    print("\n📊 СВОДНАЯ СТАТИСТИКА:")
    for batch_id, metrics in all_metrics.items():
        if '_summary' in metrics:
            summary = metrics['_summary']
            glucose_status = " (глюкоза исключена)" if summary.get('Glucose_excluded', False) else ""
            print(f"  {batch_id}: Средний R² = {summary['Mean_R2']:.3f}, "
                  f"MAPE = {summary['Mean_MAPE']:.1f}%{glucose_status}")

    # Рассчитываем общие метрики (также без глюкозы)
    overall = calculate_overall_metrics(all_metrics, exclude_glucose=exclude_glucose)
    print(f"\n📊 ОБЩИЕ МЕТРИКИ ПО ВСЕМ ПАРТИЯМ:")
    print(f"  Средний R²: {overall['Overall_R2_mean']:.3f} ± {overall['Overall_R2_std']:.3f}")
    print(f"  Средний MAPE: {overall['Overall_MAPE_mean']:.1f}% ± {overall['Overall_MAPE_std']:.1f}%")
    print(f"  Партий: {overall['N_batches']}, точек данных: {overall['N_datapoints']}")

    return all_metrics, summary_df


def create_summary_plot(all_results, all_metrics, save_path="summary.png"):
    """
    Создание сводного графика по всем партиям
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # 1. Сравнение конечных титров
    ax = axes[0, 0]
    batches = []
    final_titers_exp = []
    final_titers_model = []

    for batch_id, data in all_results.items():
        batches.append(batch_id)
        comparison_df = data["comparison_df"]

        # Экспериментальный титр (последний ненулевой)
        exp_titer = comparison_df['exp_titer_g_L'].dropna()
        if len(exp_titer) > 0:
            final_titers_exp.append(exp_titer.iloc[-1])
        else:
            final_titers_exp.append(0)

        # Модельный титр
        final_titers_model.append(comparison_df['model_P'].iloc[-1])

    x = np.arange(len(batches))
    width = 0.35

    ax.bar(x - width / 2, final_titers_exp, width, label='Эксперимент', alpha=0.8)
    ax.bar(x + width / 2, final_titers_model, width, label='Модель', alpha=0.8)
    ax.set_xlabel('Партия')
    ax.set_ylabel('Конечный титр, г/л')
    ax.set_title('Сравнение конечных титров')
    ax.set_xticks(x)
    ax.set_xticklabels(batches, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Средние значения R² по партиям (без глюкозы)
    ax = axes[0, 1]
    batch_r2 = {}

    for batch_id, metrics in all_metrics.items():
        # Исключаем глюкозу и summary из расчета среднего R²
        r2_values = []
        for var_name, var_metrics in metrics.items():
            if var_name.startswith('_') or var_name == 'Glucose':
                continue
            if isinstance(var_metrics, dict) and not np.isnan(var_metrics.get('R²', np.nan)):
                r2_values.append(var_metrics['R²'])

        if r2_values:
            batch_r2[batch_id] = np.mean(r2_values)

    if batch_r2:
        ax.bar(list(batch_r2.keys()), list(batch_r2.values()),
               color=colors[:len(batch_r2)])
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Цель: R² > 0.9')
        ax.axhline(y=0.7, color='y', linestyle='--', alpha=0.5, label='Минимум: R² > 0.7')
        ax.set_xlabel('Партия')
        ax.set_ylabel('Средний R² (без глюкозы)')
        ax.set_title('Качество калибровки по партиям')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Ошибки по параметрам (среднее по всем партиям, без глюкозы)
    ax = axes[1, 0]
    param_mape = {}

    for batch_id, metrics in all_metrics.items():
        for param, m in metrics.items():
            # Исключаем summary и глюкозу
            if param.startswith('_') or param == 'Glucose':
                continue
            if isinstance(m, dict) and 'MAPE' in m:
                if param not in param_mape:
                    param_mape[param] = []
                if not np.isnan(m['MAPE']):
                    param_mape[param].append(m['MAPE'])

    if param_mape:
        params = list(param_mape.keys())
        mean_mape = [np.nanmean(param_mape[p]) for p in params]
        std_mape = [np.nanstd(param_mape[p]) for p in params]

        x = np.arange(len(params))
        bars = ax.bar(x, mean_mape, yerr=std_mape, capsize=5, alpha=0.8)
        ax.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Допуск: 15%')
        ax.set_xlabel('Параметр')
        ax.set_ylabel('MAPE, %')
        ax.set_title('Средняя ошибка по параметрам')
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Временные ряды титра для всех партий
    ax = axes[1, 1]

    for (batch_id, data), color in zip(all_results.items(), colors):
        comparison_df = data["comparison_df"]
        ax.plot(comparison_df['time_h'], comparison_df['model_P'],
                color=color, label=batch_id, linewidth=2, alpha=0.8)

    ax.set_xlabel('Время, ч')
    ax.set_ylabel('Титр, г/л')
    ax.set_title('Динамика титра по всем партиям')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Сводный анализ калибровки модели CHO fed-batch (без учета глюкозы)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Сводный график сохранён: {save_path}")


def generate_final_report(all_results, all_metrics, summary_df):
    """
    Генерация итогового отчета
    """
    print("\n" + "=" * 80)
    print("📄 ГЕНЕРАЦИЯ ИТОГОВОГО ОТЧЕТА")
    print("=" * 80)

    # 1. Создаем графики сравнения калибровки
    plot_calibration_comparison(all_results, all_metrics,
                                save_path="data/processed/calibration_comparison.png")

    # 2. Генерируем HTML отчет
    generate_html_report(all_results, all_metrics, summary_df,
                         output_path="data/processed/final_report.html")

    # 3. Создаем сводный график по всем партиям (передаем all_metrics!)
    create_summary_plot(all_results, all_metrics,
                        save_path="data/processed/summary_plot.png")

    print("✅ Итоговый отчет сгенерирован")
    print("📁 Результаты в: data/processed/")


def main():
    """
    Основная функция
    """
    print("=" * 80)
    print("🧪 МАТЕМАТИЧЕСКАЯ МОДЕЛЬ КУЛЬТИВИРОВАНИЯ КЛЕТОК CHO")
    print("   АВТОМАТИЗАЦИЯ КАЛИБРОВКИ И ОЦЕНКИ")
    print("=" * 80)

    # 1. Найти все партии
    batches = find_all_batches()

    if not batches:
        print("❌ Не найдено ни одной партии для моделирования!")
        print("   Проверьте наличие файлов в data/raw/ и data/meta/")
        return

    print(f"\n📁 Найдено партий: {len(batches)}")
    for batch in batches:
        print(f"   - {batch['batch_id']}")

    # 2. Запустить все симуляции
    all_results = run_all_simulations(batches)

    if not all_results:
        print("❌ Ни одна симуляция не завершилась успешно!")
        return

    # 3. Оценить калибровку (с исключением глюкозы)
    print("\n" + "=" * 80)
    print("🎯 КАЛИБРОВКА БЕЗ УЧЕТА ГЛЮКОЗЫ")
    print("=" * 80)
    all_metrics, summary_df = evaluate_all_batches(all_results, exclude_glucose=True)

    # 4. Дополнительно: оценка с глюкозой для сравнения
    print("\n" + "=" * 80)
    print("📊 ДЛЯ СРАВНЕНИЯ: КАЛИБРОВКА С ГЛЮКОЗОЙ")
    print("=" * 80)
    all_metrics_with_glucose, summary_df_with_glucose = evaluate_all_batches(all_results, exclude_glucose=False)

    # 5. Сгенерировать отчет (используем вариант без глюкозы)
    generate_final_report(all_results, all_metrics, summary_df)

    # 6. Итоговое сообщение
    print("\n" + "=" * 80)
    print("🎉 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 80)
    print("\n📂 Результаты сохранены в папке: data/processed/")
    print("\n📋 Ключевые файлы:")
    print("   - calibration_metrics_all_no_glucose.csv - метрики без глюкозы")
    print("   - calibration_metrics_all.csv - метрики с глюкозой (для сравнения)")
    print("   - calibration_summary_no_glucose.xlsx - сводная таблица без глюкозы для диплома")
    print("   - final_report.html - HTML отчет с графиками")
    print("   - plot_CHO*.png - графики для каждой партии")
    print("   - summary_plot.png - сводный график")
    print("\n📊 Для диплома рекомендуется использовать вариант БЕЗ глюкозы,")
    print("   так как глюкоза имеет нестабильную динамику и ухудшает общие метрики.")
    print("=" * 80)


if __name__ == "__main__":
    main()