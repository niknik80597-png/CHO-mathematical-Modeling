#!/usr/bin/env python3
"""
Главный файл для гибридного моделирования CHO клеток
Выбор между механистической и гибридной (ML) моделью
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import argparse


def print_banner():
    """Вывод баннера программы"""
    banner = f"""
{'=' * 80}
  ГИБРИДНАЯ МОДЕЛЬ КУЛЬТИВИРОВАНИЯ КЛЕТОК CHO
  Механистическая модель + Machine Learning коррекция
{'=' * 80}
"""
    print(banner)


def print_menu():
    """Вывод меню выбора"""
    menu = f"""
{'=' * 80}
МЕНЮ ВЫБОРА РЕЖИМА МОДЕЛИРОВАНИЯ
{'=' * 80}

1. Механистическая модель (только физические уравнения)
2. Гибридная модель (механистика + ML коррекция)
3. Обучение ML модели (Random Forest)
4. Сравнение моделей (механистика vs гибридная)
5. Статистика и метрики
6. Выход
{'=' * 80}
"""
    print(menu)


def get_user_choice():
    """Получение выбора пользователя"""
    while True:
        try:
            choice = input("\nВыберите опцию (1-6): ")
            if choice.isdigit() and 1 <= int(choice) <= 6:
                return int(choice)
            else:
                print("Ошибка: введите число от 1 до 6")
        except KeyboardInterrupt:
            print("\nДо свидания!")
            sys.exit(0)
        except Exception as e:
            print(f"Ошибка: {str(e)}")


def run_mechanistic_model():
    """Запуск чисто механистической модели"""
    print(f"\n{'=' * 80}")
    print("ЗАПУСК МЕХАНИСТИЧЕСКОЙ МОДЕЛИ")
    print(f"{'=' * 80}")

    # Проверяем наличие необходимых файлов
    try:
        # Импортируем основной симулятор
        from run_all_batches import main as run_all

        print("Поиск доступных партий...")

        # Запускаем механистическую модель
        run_all()

        print("\nМеханистическая модель успешно запущена!")
        print("Результаты сохранены в: data/processed/")

    except ImportError as e:
        print(f"Ошибка импорта: {str(e)}")
        print("Убедитесь, что все модули установлены")
        return False
    except Exception as e:
        print(f"Ошибка выполнения: {str(e)}")
        return False

    return True


def run_hybrid_model():
    """Запуск гибридной модели (механистика + ML)"""
    print(f"\n{'=' * 80}")
    print("ЗАПУСК ГИБРИДНОЙ МОДЕЛИ (Механистика + ML)")
    print(f"{'=' * 80}")

    try:
        # Проверяем, обучена ли ML модель
        ml_model_path = "ml_models/random_forest_model.pkl"
        if not os.path.exists(ml_model_path):
            print("ML модель не найдена! Сначала обучите модель (опция 3)")

            # Спросим, хотим ли обучить сейчас
            train_now = input("Обучить ML модель сейчас? (y/n): ")
            if train_now.lower() == 'y':
                if not train_ml_model():
                    return False
            else:
                return False

        # Импортируем гибридный симулятор
        print("Загрузка гибридного симулятора...")
        from hybrid_simulator import run_all_hybrid_simulations

        # Спрашиваем пользователя, какие партии запускать
        print("\nВЫБОР ПАРТИЙ ДЛЯ ГИБРИДНОГО МОДЕЛИРОВАНИЯ:")
        print("   a) Все партии")
        print("   b) Выбрать конкретные партии")
        print("   c) Только проблемные партии (CHO01, CHO02)")

        choice = input("Выберите опцию (a/b/c): ").lower()

        if choice == 'a':
            # Запускаем гибридную симуляцию для всех партий
            print("Запуск для всех партий...")
            all_results = run_all_hybrid_simulations(use_ml_correction=True)

        elif choice == 'b':
            # Пользователь выбирает конкретные партии
            print("Доступные партии: CHO01, CHO02, CHO09")
            batch_input = input("Введите номера партий через запятую (например: 01,02): ")

            batch_numbers = [b.strip() for b in batch_input.split(',')]
            all_results = {}

            for batch_num in batch_numbers:
                batch_id = f"CHO{batch_num}"
                print(f"\nЗапуск гибридной симуляции для {batch_id}...")

                # Запускаем для конкретной партии
                from hybrid_simulator import run_hybrid_simulation

                csv_path = f"data/raw/batch_{batch_id}.csv"
                meta_path = f"data/meta/batch_{batch_id}.json"
                output_path = f"data/hybrid_results/simulation_{batch_id}"

                if not os.path.exists(csv_path):
                    print(f"Файл не найден: {csv_path}")
                    continue
                if not os.path.exists(meta_path):
                    print(f"Файл не найден: {meta_path}")
                    continue

                results = run_hybrid_simulation(
                    csv_path=csv_path,
                    meta_path=meta_path,
                    output_path=output_path,
                    batch_id=batch_id,
                    use_ml_correction=True
                )

                if results:
                    all_results[batch_id] = results

        elif choice == 'c':
            # Только проблемные партии
            print("Запуск для проблемных партий (CHO01, CHO02)...")
            from hybrid_simulator import run_hybrid_simulation

            all_results = {}
            problem_batches = ["CHO01", "CHO02"]

            for batch_id in problem_batches:
                print(f"\nЗапуск гибридной симуляции для {batch_id}...")

                csv_path = f"data/raw/batch_{batch_id}.csv"
                meta_path = f"data/meta/batch_{batch_id}.json"
                output_path = f"data/hybrid_results/simulation_{batch_id}"

                if not os.path.exists(csv_path):
                    print(f"Файл не найден: {csv_path}")
                    continue
                if not os.path.exists(meta_path):
                    print(f"Файл не найден: {meta_path}")
                    continue

                results = run_hybrid_simulation(
                    csv_path=csv_path,
                    meta_path=meta_path,
                    output_path=output_path,
                    batch_id=batch_id,
                    use_ml_correction=True
                )

                if results:
                    all_results[batch_id] = results

        else:
            print("Неверный выбор!")
            return False

        # Показываем результаты
        if all_results:
            print("\nГибридная модель успешно запущена!")
            print("Результаты сохранены в: data/hybrid_results/")

            # Вычисляем и показываем метрики
            from hybrid_simulator import calculate_hybrid_metrics, save_hybrid_results
            metrics = calculate_hybrid_metrics(all_results)
            save_hybrid_results(all_results, metrics)

            # Показываем ключевые метрики
            if metrics and 'summary' in metrics:
                summary = metrics['summary']
                print("\nКЛЮЧЕВЫЕ МЕТРИКИ ГИБРИДНОЙ МОДЕЛИ:")
                print(f"  Количество партий: {summary.get('total_batches', 0)}")
                print(f"  Средний R²: {summary.get('mean_r2', 0):.3f}")
                print(f"  Средний MAPE: {summary.get('mean_mape', 0):.1f}%")

                # Сравнение с механистической моделью
                try:
                    from evaluation.metrics import calculate_overall_metrics
                    mech_metrics_path = "data/processed/calibration_metrics_all_no_glucose.csv"
                    if os.path.exists(mech_metrics_path):
                        mech_df = pd.read_csv(mech_metrics_path)
                        mech_r2_values = pd.to_numeric(mech_df['R²'], errors='coerce')
                        mech_mean_r2 = mech_r2_values.mean()

                        improvement = summary.get('mean_r2', 0) - mech_mean_r2
                        print(f"  Улучшение R²: {improvement:.3f} ({improvement * 100:.1f}%)")
                except:
                    pass
        else:
            print("Нет результатов для отображения")

        return True

    except Exception as e:
        print(f"Ошибка выполнения: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def train_ml_model():
    """Обучение ML модели (Random Forest)"""
    print(f"\n{'=' * 80}")
    print("ОБУЧЕНИЕ ML МОДЕЛИ (Random Forest)")
    print(f"{'=' * 80}")

    try:
        # Импортируем ML тренер
        from ml_trainer import train_random_forest, evaluate_ml_model

        print("Загрузка данных для обучения...")

        # Обучение модели
        model, metrics = train_random_forest()

        if model is None:
            print("Ошибка при обучении модели!")
            return False

        print("\nML модель успешно обучена!")

        # Вывод метрик
        print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        print(f"  Точность (R²) на тренировочных данных: {metrics.get('train_r2', 0):.3f}")
        print(f"  Точность (R²) на тестовых данных: {metrics.get('test_r2', 0):.3f}")
        print(f"  Средняя абсолютная ошибка (MAE): {metrics.get('mae', 0):.4f}")
        print(f"  Важность признаков:")

        if 'feature_importance' in metrics:
            for feature, importance in metrics['feature_importance'].items():
                print(f"    - {feature}: {importance:.3f}")

        # Оценка модели на всех данных
        print("\nОценка ML модели на всех данных...")
        evaluation_results = evaluate_ml_model(model)

        if evaluation_results:
            print("\nОценка завершена!")
            print(f"  Средняя ошибка предсказания: {evaluation_results.get('mean_error', 0):.4f}")
            print(f"  Стандартное отклонение: {evaluation_results.get('std_error', 0):.4f}")

        # Сохранение информации о модели
        model_info = {
            "model_type": "Random Forest",
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "features_used": list(metrics.get('feature_importance', {}).keys()),
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        }

        # Сохраняем информацию о модели
        os.makedirs("ml_models", exist_ok=True)
        with open("ml_models/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        print("\nМодель сохранена в: ml_models/")

        return True

    except Exception as e:
        print(f"Ошибка при обучении ML модели: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def compare_models():
    """Сравнение механистической и гибридной моделей"""
    print(f"\n{'=' * 80}")
    print("СРАВНЕНИЕ МОДЕЛЕЙ (Механистическая vs Гибридная)")
    print(f"{'=' * 80}")

    try:
        from model_comparison import compare_model_performance

        print("Загрузка результатов обеих моделей...")

        results = compare_model_performance()

        if not results:
            print("Нет данных для сравнения!")
            print("   Запустите сначала обе модели (опции 1 и 2)")
            return False

        print("\nСравнение выполнено!")

        # Вывод результатов сравнения
        print(f"\n{'=' * 80}")
        print("СВОДКА СРАВНЕНИЯ:")
        print(f"{'=' * 80}")

        print("\nМеханистическая модель:")
        print(f"  Средний R²: {results.get('mech_mean_r2', 0):.3f}")
        print(f"  Средний MAPE: {results.get('mech_mean_mape', 0):.1f}%")

        print("\nГибридная модель (с ML):")
        print(f"  Средний R²: {results.get('hybrid_mean_r2', 0):.3f}")
        print(f"  Средний MAPE: {results.get('hybrid_mean_mape', 0):.1f}%")

        print(f"\n{'=' * 80}")
        print("УЛУЧШЕНИЕ С ГИБРИДНОЙ МОДЕЛЬЮ:")
        print(f"{'=' * 80}")

        r2_improvement = results.get('r2_improvement', 0)
        mape_improvement = results.get('mape_improvement', 0)

        print(f"  R² улучшился на: {r2_improvement:.3f} ({r2_improvement * 100:.1f}%)")
        print(f"  MAPE уменьшился на: {mape_improvement:.1f}%")

        # Определяем, какая модель лучше
        if r2_improvement > 0.05:  # Улучшение более 5%
            print("\nГибридная модель значительно лучше!")
        elif r2_improvement > 0.02:
            print("\nГибридная модель немного лучше")
        else:
            print("\nМодели показывают схожие результаты")

        # Сохранение результатов сравнения
        comparison_path = "data/comparison/results_comparison.json"
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nРезультаты сравнения сохранены: {comparison_path}")

        # Создание графиков сравнения
        print("\nСоздание графиков сравнения...")
        try:
            from model_comparison import create_comparison_plots
            create_comparison_plots(results)
            print("Графики сохранены в: data/comparison/plots/")
        except Exception as e:
            print(f"Не удалось создать графики: {str(e)}")

        return True

    except ImportError as e:
        print(f"Ошибка импорта: {str(e)}")
        return False
    except Exception as e:
        print(f"Ошибка при сравнении моделей: {str(e)}")
        return False


def show_statistics():
    """Показ статистики и метрик"""
    print(f"\n{'=' * 80}")
    print("СТАТИСТИКА И МЕТРИКИ")
    print(f"{'=' * 80}")

    # Проверяем существующие файлы
    stats_files = [
        "data/processed/calibration_metrics_all_no_glucose.csv",
        "data/processed/calibration_summary_no_glucose.xlsx",
        "data/hybrid_results/hybrid_metrics.json",
        "ml_models/model_info.json"
    ]

    existing_files = []
    for file in stats_files:
        if os.path.exists(file):
            existing_files.append(file)

    if not existing_files:
        print("Файлы статистики не найдены!")
        print("   Запустите сначала моделирование (опции 1 или 2)")
        return

    print("\nНайдены файлы статистики:")
    for file in existing_files:
        print(f"  • {file}")

    # Показываем сводную информацию
    try:
        if os.path.exists("ml_models/model_info.json"):
            with open("ml_models/model_info.json", "r") as f:
                model_info = json.load(f)

            print("\nИНФОРМАЦИЯ О ML МОДЕЛИ:")
            print(f"  Тип модели: {model_info.get('model_type', 'N/A')}")
            print(f"  Дата обучения: {model_info.get('training_date', 'N/A')}")

            metrics = model_info.get('metrics', {})
            if metrics:
                print(f"  Точность (R²): {metrics.get('test_r2', 0):.3f}")
                print(f"  Количество признаков: {len(model_info.get('features_used', []))}")

        # Показываем метрики калибровки
        if os.path.exists("data/processed/calibration_metrics_all_no_glucose.csv"):
            df = pd.read_csv("data/processed/calibration_metrics_all_no_glucose.csv")

            print("\nМЕТРИКИ КАЛИБРОВКИ:")
            print(f"  Всего партий: {df['Партия'].nunique()}")
            print(f"  Всего параметров: {df['Параметр'].nunique()}")

            # Средние метрики
            print("\n  Средние значения по всем партиям:")
            print(f"    Средний R²: {df['R²'].apply(lambda x: float(x) if x != 'NaN' else 0).mean():.3f}")

            # Лучшие и худшие параметры
            best_params = df.groupby('Параметр')['R²'].apply(
                lambda x: pd.to_numeric(x, errors='coerce').mean()
            ).sort_values(ascending=False)

            print("\n  Лучшие параметры (по R²):")
            for i, (param, r2) in enumerate(best_params.head(3).items()):
                print(f"    {i + 1}. {param}: {r2:.3f}")

            print("\n  Худшие параметры (по R²):")
            for i, (param, r2) in enumerate(best_params.tail(3).items()):
                print(f"    {i + 1}. {param}: {r2:.3f}")

    except Exception as e:
        print(f"Ошибка при чтении статистики: {str(e)}")

    print("\nДля подробной статистики откройте соответствующие файлы.")


def main():
    """Главная функция"""
    print_banner()

    while True:
        print_menu()
        choice = get_user_choice()

        if choice == 1:
            run_mechanistic_model()
        elif choice == 2:
            run_hybrid_model()
        elif choice == 3:
            train_ml_model()
        elif choice == 4:
            compare_models()
        elif choice == 5:
            show_statistics()
        elif choice == 6:
            print(f"\n{'=' * 80}")
            print("Спасибо за использование программы!")
            print("   До свидания!")
            print(f"{'=' * 80}")
            break

        # Пауза между действиями
        if choice != 6:
            input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='Гибридная модель культивирования CHO клеток')
    parser.add_argument('--mode', type=str, choices=['mech', 'hybrid', 'train', 'compare'],
                        help='Режим запуска (mech, hybrid, train, compare)')

    args = parser.parse_args()

    if args.mode:
        # Режим командной строки
        if args.mode == 'mech':
            run_mechanistic_model()
        elif args.mode == 'hybrid':
            run_hybrid_model()
        elif args.mode == 'train':
            train_ml_model()
        elif args.mode == 'compare':
            compare_models()
    else:
        # Интерактивный режим с меню
        main()