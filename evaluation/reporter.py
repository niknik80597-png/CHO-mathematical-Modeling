"""
Модуль для генерации отчетов
"""

import pandas as pd
import numpy as np
from datetime import datetime


def generate_html_report(all_results, all_metrics, summary_df, output_path="report.html"):
    """
    Генерация HTML отчета с результатами

    Parameters:
    -----------
    all_results : dict
        Результаты всех симуляций
    all_metrics : dict
        Метрики калибровки
    summary_df : pd.DataFrame
        Сводная таблица метрик
    output_path : str
        Путь для сохранения HTML отчета
    """

    # Рассчитываем общие метрики
    overall_metrics = calculate_overall_metrics(all_metrics)

    # Создаем HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Отчет по калибровке модели CHO fed-batch</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ background-color: #3498db; color: white; padding: 20px; border-radius: 10px; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .metrics {{ background-color: #e8f4fc; padding: 15px; border-radius: 10px; }}
            .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .table th {{ background-color: #3498db; color: white; }}
            .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .good {{ color: #27ae60; font-weight: bold; }}
            .warning {{ color: #f39c12; font-weight: bold; }}
            .bad {{ color: #e74c3c; font-weight: bold; }}
            .images {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
            .image-container {{ flex: 1 1 300px; }}
            .image-container img {{ width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Отчет по калибровке математической модели CHO fed-batch</h1>
            <p>Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Всего партий: {len(all_results)}</p>
        </div>

        <div class="summary">
            <h2>📈 Общие результаты</h2>
            <div class="metrics">
    """

    # Добавляем общие метрики
    if 'Overall_R2_mean' in overall_metrics and not np.isnan(overall_metrics['Overall_R2_mean']):
        r2_class = "good" if overall_metrics['Overall_R2_mean'] > 0.8 else "warning" if overall_metrics[
                                                                                            'Overall_R2_mean'] > 0.6 else "bad"
        mape_class = "good" if overall_metrics['Overall_MAPE_mean'] < 15 else "warning" if overall_metrics[
                                                                                               'Overall_MAPE_mean'] < 25 else "bad"

        html_content += f"""
                <p><strong>Средний R² по всем партиям и параметрам:</strong> 
                   <span class="{r2_class}">{overall_metrics['Overall_R2_mean']:.3f} ± {overall_metrics['Overall_R2_std']:.3f}</span></p>
                <p><strong>Средняя относительная ошибка (MAPE):</strong> 
                   <span class="{mape_class}">{overall_metrics['Overall_MAPE_mean']:.1f}% ± {overall_metrics['Overall_MAPE_std']:.1f}%</span></p>
                <p><strong>Всего точек данных:</strong> {overall_metrics['N_datapoints']}</p>
        """
    else:
        html_content += "<p><strong>Недостаточно данных для расчета общих метрик</strong></p>"

    html_content += """
            </div>
        </div>

        <h2>📋 Детальные метрики по партиям</h2>
    """

    # Добавляем сводную таблицу
    if not summary_df.empty:
        html_content += summary_df.to_html(classes='table', index=False, escape=False)

    # Добавляем раздел по каждой партии
    html_content += "<h2>🔍 Детали по партиям</h2>"

    for batch_id, data in all_results.items():
        html_content += f"""
        <div class="summary">
            <h3>Партия {batch_id}</h3>
            <p><strong>Конечный титр (модель):</strong> {data['results_df']['P'].iloc[-1]:.2f} г/л</p>
            <p><strong>Пиковая Xv:</strong> {data['results_df']['Xv'].max():.2f} ×10⁶ кл/мл</p>
            <p><strong>Финальная жизнеспособность:</strong> {data['results_df']['viability'].iloc[-1]:.2%}</p>

            <div class="images">
                <div class="image-container">
                    <p><strong>График моделирования:</strong></p>
                    <img src="plot_{batch_id}.png" alt="График {batch_id}">
                </div>
        """

        # Добавляем метрики для этой партии
        if batch_id in all_metrics:
            batch_metrics = all_metrics[batch_id]
            if '_summary' in batch_metrics:
                html_content += f"""
                <div class="metrics">
                    <p><strong>Средний R²:</strong> {batch_metrics['_summary']['Mean_R2']:.3f}</p>
                    <p><strong>Средний MAPE:</strong> {batch_metrics['_summary']['Mean_MAPE']:.1f}%</p>
                </div>
                """

        html_content += """
            </div>
        </div>
        """

    # Добавляем рекомендации
    html_content += """
        <div class="summary">
            <h2>🎯 Рекомендации и выводы</h2>
            <div class="metrics">
                <h3>Критерии качества калибровки:</h3>
                <ul>
                    <li><span class="good">Отличная калибровка:</span> R² > 0.85, MAPE < 10%</li>
                    <li><span class="warning">Хорошая калибровка:</span> R² = 0.70-0.85, MAPE = 10-20%</li>
                    <li><span class="bad">Требуется доработка:</span> R² < 0.70, MAPE > 20%</li>
                </ul>

                <h3>Рекомендации по улучшению модели:</h3>
                <ol>
                    <li>Для параметров с низким R² провести дополнительную калибровку</li>
                    <li>Уточнить кинетические параметры для проблемных партий</li>
                    <li>Рассмотреть влияние ML-коррекции для улучшения предсказания титра</li>
                    <li>Провести анализ чувствительности параметров</li>
                </ol>
            </div>
        </div>

        <div class="summary">
            <h2>📁 Файлы результатов</h2>
            <ul>
                <li><strong>data/processed/calibration_metrics_all.csv</strong> - все метрики калибровки</li>
                <li><strong>data/processed/calibration_summary.xlsx</strong> - сводная таблица для диплома</li>
                <li><strong>data/processed/simulation_*_states.csv</strong> - состояния модели для каждой партии</li>
                <li><strong>data/processed/simulation_*_rates.csv</strong> - скорости процессов</li>
                <li><strong>data/processed/simulation_*_comparison.csv</strong> - сравнение с экспериментом</li>
                <li><strong>data/processed/plot_*.png</strong> - графики для каждой партии</li>
                <li><strong>data/processed/summary_plot.png</strong> - сводный график</li>
            </ul>
        </div>

        <footer style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 0.9em;">
            <p>Математическая модель культивирования клеток CHO в режиме fed-batch</p>
            <p>© Воронков Н.Н., {datetime.now().year}</p>
        </footer>
    </body>
    </html>
    """

    # Сохраняем HTML файл
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"📄 HTML отчет сохранён: {output_path}")


def calculate_overall_metrics(all_metrics, exclude_glucose=True):
    """Вспомогательная функция для расчета общих метрик"""
    all_r2 = []
    all_mape = []

    for batch_metrics in all_metrics.values():
        for var_name, var_metrics in batch_metrics.items():
            if var_name.startswith('_'):
                continue

            # Исключаем глюкозу из общих метрик
            if exclude_glucose and var_name == 'Glucose':
                continue

            if not np.isnan(var_metrics['R²']):
                all_r2.append(var_metrics['R²'])

            if not np.isnan(var_metrics['MAPE']):
                all_mape.append(var_metrics['MAPE'])

    if all_r2 and all_mape:
        return {
            'Overall_R2_mean': np.mean(all_r2),
            'Overall_R2_std': np.std(all_r2),
            'Overall_MAPE_mean': np.mean(all_mape),
            'Overall_MAPE_std': np.std(all_mape),
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