
"""
Гибридный симулятор: механистическая модель + ML коррекция
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
import glob

# Импортируем механистическую модель
from main_simulation import run_simulation, create_comparison_df
from model.simulator import simulation_step


class HybridSimulator:
    """Гибридный симулятор с ML коррекцией"""

    def __init__(self, use_ml_correction=True):
        self.use_ml_correction = use_ml_correction
        self.ml_model = None
        self.scaler = None
        self.feature_names = None

        if use_ml_correction:
            self.load_ml_model()

    def load_ml_model(self):
        """Загрузка обученной ML модели"""
        try:
            model_path = "ml_models/random_forest_model.pkl"
            scaler_path = "ml_models/scaler.pkl"

            if not os.path.exists(model_path):
                print("⚠️  ML модель не найдена! Будет использоваться только механистическая модель")
                self.use_ml_correction = False
                return

            with open(model_path, "rb") as f:
                self.ml_model = pickle.load(f)

            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            # Загружаем информацию о признаках
            info_path = "ml_models/model_info.json"
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    model_info = json.load(f)
                self.feature_names = model_info.get('features_used', [])

            print(f"✅ ML модель загружена: {self.ml_model.__class__.__name__}")
            print(f"   Количество деревьев: {self.ml_model.n_estimators}")

        except Exception as e:
            print(f"❌ Ошибка при загрузке ML модели: {str(e)}")
            self.use_ml_correction = False

    def prepare_features_for_prediction(self, state, rates, t, t_shift, params):
        """
        Подготовка признаков для ML модели
        """
        features = {}

        # 1. Основные состояния
        features['time_h'] = t
        features['TCD'] = state.get('TCD', 0)
        features['glucose'] = state.get('G', 0)
        features['lactate'] = state.get('Lac', 0)
        features['ammonium'] = state.get('NH4', 0)
        features['model_titer'] = state.get('P', 0)

        # 2. Производные признаки
        features['time_norm'] = t / 240 if t > 0 else 0  # Нормализация по общему времени
        features['glucose_lactate_ratio'] = features['glucose'] / (features['lactate'] + 0.1)
        features['metabolic_quotient'] = features['lactate'] / (features['glucose'] + 0.1)

        # 3. Кинетические параметры
        features['mu'] = rates.get('mu', 0)
        features['qP'] = rates.get('qP', 0)
        features['qG'] = rates.get('qG', 0)

        # 4. Дополнительные признаки
        features['TCD_squared'] = features['TCD'] ** 2
        features['glucose_squared'] = features['glucose'] ** 2
        features['TCD_times_time'] = features['TCD'] * features['time_norm']

        # 5. Температурная фаза
        features['temperature_phase'] = 0 if t < t_shift else 1

        # 6. Время после сдвига
        features['time_post_shift'] = max(0, t - t_shift)

        return features

    def get_ml_correction(self, features_dict):
        """
        Получение ML коррекции для предсказания
        """
        if not self.use_ml_correction or self.ml_model is None:
            return 0.0

        try:
            # Преобразуем словарь признаков в массив
            if self.feature_names:
                # Используем только те признаки, которые были при обучении
                features_array = []
                for feature_name in self.feature_names:
                    if feature_name in features_dict:
                        features_array.append(features_dict[feature_name])
                    else:
                        features_array.append(0.0)  # Заполняем нулями отсутствующие признаки
            else:
                # Если нет информации о признаках, используем все
                features_array = list(features_dict.values())

            # Масштабирование признаков
            if self.scaler:
                features_scaled = self.scaler.transform([features_array])
            else:
                features_scaled = [features_array]

            # Предсказание коррекции
            correction = self.ml_model.predict(features_scaled)[0]

            # Ограничиваем коррекцию разумными пределами
            max_correction = 0.5  # Максимальная коррекция 50%
            correction = np.clip(correction, -max_correction, max_correction)

            return correction

        except Exception as e:
            print(f"⚠️  Ошибка при получении ML коррекции: {str(e)}")
            return 0.0

    def hybrid_simulation_step(self, state, inputs, params, dt, t_shift, temp_coeffs, batch_id=None):
        """
        Гибридный шаг симуляции: механистика + ML коррекция
        """
        # Механистический шаг
        new_state, rates = simulation_step(state, inputs, params, dt, t_shift, temp_coeffs)

        # Если включена ML коррекция и есть данные о титре
        if self.use_ml_correction and 'P' in new_state:
            # Подготавливаем признаки для ML
            features = self.prepare_features_for_prediction(
                new_state, rates, state['time_h'], t_shift, params
            )

            # Получаем ML коррекцию для титра
            ml_correction = self.get_ml_correction(features)

            # Применяем коррекцию к титру
            original_titer = new_state['P']
            corrected_titer = original_titer * (1 + ml_correction)

            # Ограничиваем титр снизу нулем
            new_state['P'] = max(corrected_titer, 0.0)

            # Сохраняем информацию о коррекции для анализа
            new_state['ml_correction'] = ml_correction
            new_state['ml_correction_abs'] = corrected_titer - original_titer

        return new_state, rates


def run_hybrid_simulation(csv_path=None, meta_path=None, output_path=None,
                          batch_id=None, use_ml_correction=True):
    """
    Запуск гибридной симуляции для одной партии
    """
    print(f"\n🤖 ЗАПУСК ГИБРИДНОЙ СИМУЛЯЦИИ: {batch_id}")

    # Загружаем экспериментальные данные
    df_exp = pd.read_csv(csv_path)

    # Создаем гибридный симулятор
    simulator = HybridSimulator(use_ml_correction=use_ml_correction)

    # Загружаем параметры из JSON
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Начальные условия
    initial = meta["initial_conditions"]
    state = {
        "time_h": 0,
        "V": initial["V"],
        "TCD": initial["TCD"],
        "Xv": initial["TCD"] * initial["Viab"],
        "G": initial["G"],
        "Lac": initial["Lac"],
        "NH4": initial["NH4"],
        "P": initial["P"],
        "viability": initial["Viab"]
    }

    params = meta["kinetics_parameters"]
    t_shift = meta["process_parameters"]["time_shift_h"]
    dt = meta.get("time_step_h", meta["process_time"]["time_step_h"])
    total_duration = meta["process_time"]["total_duration_h"]
    temp_coeffs = meta["temperature_coefficients"]

    # Подготовка интерполяторов для подачи
    time_points = df_exp['time_h'].values

    from scipy.interpolate import interp1d
    feed_glucose_interp = interp1d(
        time_points, df_exp['feed_glucose_gph'].values,
        bounds_error=False, fill_value=0.0
    )
    feed_other_interp = interp1d(
        time_points, df_exp['feed_other_gph'].values,
        bounds_error=False, fill_value=0.0
    )

    # Цикл гибридной симуляции
    results = []
    rates_history = []
    ml_corrections = []

    simulation_times = np.arange(0, total_duration + dt, dt)

    for i, t in enumerate(simulation_times):
        # Получаем входные данные
        F_glc = float(feed_glucose_interp(t))
        F_other = float(feed_other_interp(t))

        # Сохраняем текущее состояние
        results.append(state.copy())

        # Выполняем гибридный шаг
        state, rates = simulator.hybrid_simulation_step(
            state=state,
            inputs=(F_glc, F_other),
            params=params,
            dt=dt,
            t_shift=t_shift,
            temp_coeffs=temp_coeffs,
            batch_id=batch_id
        )

        # Сохраняем скорости и ML коррекции
        rates["time_h"] = t
        rates_history.append(rates)

        if 'ml_correction' in state:
            ml_corrections.append({
                'time_h': t,
                'correction': state['ml_correction'],
                'correction_abs': state.get('ml_correction_abs', 0),
                'original_titer': state.get('P', 0) / (1 + state['ml_correction'])
                if state['ml_correction'] != 0 else state.get('P', 0)
            })

        # Прогресс
        if i % 10 == 0:
            progress = t / total_duration * 100
            print(f"  Прогресс: {progress:.1f}% (t={t:.0f} ч, P={state['P']:.2f} г/л)")

    # Создаем DataFrames
    results_df = pd.DataFrame(results)
    rates_df = pd.DataFrame(rates_history)

    if ml_corrections:
        ml_df = pd.DataFrame(ml_corrections)
    else:
        ml_df = pd.DataFrame()

    results_df["NH4_mM"] = results_df["NH4"] / 0.018
    results_df["batch_id"] = batch_id

    # Сравнение с экспериментальными данными
    comparison_df = create_comparison_df(results_df, df_exp, batch_id)

    # Добавляем информацию о ML коррекции в comparison_df
    if not ml_df.empty:
        # Интерполируем коррекции на времена сравнения
        from scipy.interpolate import interp1d
        ml_interp = interp1d(ml_df['time_h'], ml_df['correction'],
                             bounds_error=False, fill_value=0.0)
        comparison_df['ml_correction'] = ml_interp(comparison_df['time_h'])

        ml_abs_interp = interp1d(ml_df['time_h'], ml_df['correction_abs'],
                                 bounds_error=False, fill_value=0.0)
        comparison_df['ml_correction_abs'] = ml_abs_interp(comparison_df['time_h'])

    # Сохранение результатов
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        base_name = f"{output_path}_{batch_id}_hybrid"
        results_df.to_csv(f"{base_name}_states.csv", index=False)
        rates_df.to_csv(f"{base_name}_rates.csv", index=False)
        comparison_df.to_csv(f"{base_name}_comparison.csv", index=False)

        if not ml_df.empty:
            ml_df.to_csv(f"{base_name}_ml_corrections.csv", index=False)

        print(f"✅ Результаты сохранены: {base_name}_*.csv")

    print(f"\n🎯 ИТОГИ ГИБРИДНОЙ СИМУЛЯЦИИ {batch_id}:")
    print(f"   - Конечный титр: {results_df['P'].iloc[-1]:.2f} г/л")
    print(f"   - Пиковая Xv: {results_df['Xv'].max():.2f} ×10⁶ кл/мл")

    if not ml_df.empty:
        avg_correction = ml_df['correction'].mean() * 100
        print(f"   - Средняя ML коррекция: {avg_correction:.1f}%")

    return {
        'results_df': results_df,
        'rates_df': rates_df,
        'comparison_df': comparison_df,
        'ml_corrections': ml_df,
        'batch_id': batch_id
    }


def run_all_hybrid_simulations(use_ml_correction=True):
    """
    Запуск гибридной симуляции для всех партий
    """
    print(f"\n{'=' * 80}")
    print(f"🤖 ГИБРИДНОЕ МОДЕЛИРОВАНИЕ ВСЕХ ПАРТИЙ")
    print(f"{'=' * 80}")

    # Находим все партии
    data_dir = "data/raw"
    meta_dir = "data/meta"

    csv_files = glob.glob(os.path.join(data_dir, "batch_CHO*.csv"))
    all_results = {}

    for csv_path in csv_files:
        # Извлекаем номер партии
        base_name = os.path.basename(csv_path)
        batch_num = base_name.replace("batch_CHO", "").replace(".csv", "")
        batch_id = f"CHO{batch_num}"

        # Проверяем существование JSON файла
        meta_path = os.path.join(meta_dir, f"batch_CHO{batch_num}.json")

        if not os.path.exists(meta_path):
            print(f"⚠️  Пропускаем {batch_id}: нет JSON файла")
            continue

        print(f"\n📊 ПАРТИЯ: {batch_id}")
        print(f"{'-' * 40}")

        try:
            # Запускаем гибридную симуляцию
            output_prefix = f"data/hybrid_results/simulation_{batch_id}"

            results = run_hybrid_simulation(
                csv_path=csv_path,
                meta_path=meta_path,
                output_path=output_prefix,
                batch_id=batch_id,
                use_ml_correction=use_ml_correction
            )

            all_results[batch_id] = results

            print(f"✅ {batch_id}: Успешно завершено")

        except Exception as e:
            print(f"❌ {batch_id}: Ошибка - {str(e)}")
            import traceback
            traceback.print_exc()

    # Вычисляем общие метрики
    if all_results:
        metrics = calculate_hybrid_metrics(all_results)
        save_hybrid_results(all_results, metrics)

    print(f"\n{'=' * 80}")
    print(f"🎯 ГИБРИДНОЕ МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"   Успешно: {len(all_results)} партий")
    print(f"{'=' * 80}")

    return all_results


def calculate_hybrid_metrics(all_results):
    """
    Вычисление метрик для гибридной модели
    """
    from evaluation.metrics import calculate_calibration_metrics

    all_metrics = {}

    for batch_id, data in all_results.items():
        comparison_df = data['comparison_df']

        # Вычисляем метрики (без глюкозы)
        metrics = calculate_calibration_metrics(comparison_df, batch_id, exclude_glucose=True)
        all_metrics[batch_id] = metrics

    # Сводные метрики
    summary = {
        'total_batches': len(all_metrics),
        'batches': list(all_metrics.keys()),
        'mean_r2': np.mean([m['_summary']['Mean_R2'] for m in all_metrics.values()
                            if not np.isnan(m['_summary']['Mean_R2'])]),
        'mean_mape': np.mean([m['_summary']['Mean_MAPE'] for m in all_metrics.values()
                              if not np.isnan(m['_summary']['Mean_MAPE'])]),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return {
        'detailed': all_metrics,
        'summary': summary
    }


def save_hybrid_results(all_results, metrics):
    """
    Сохранение результатов гибридного моделирования
    """
    output_dir = "data/hybrid_results"
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем метрики с правильной кодировкой
    metrics_path = os.path.join(output_dir, "hybrid_metrics.json")
    try:
        with open(metrics_path, "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ Метрики сохранены: {metrics_path}")
    except Exception as e:
        print(f"⚠️  Ошибка сохранения JSON: {str(e)}")
        # Пробуем с ensure_ascii=True для ASCII-совместимости
        with open(metrics_path, "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=True)
        print(f"✅ Метрики сохранены с ASCII-кодировкой")

    # Создаем сводную таблицу
    summary_data = []

    for batch_id, data in all_results.items():
        comparison_df = data['comparison_df']
        batch_metrics = metrics['detailed'][batch_id]

        # Основные показатели
        final_titer = data['results_df']['P'].iloc[-1]
        peak_xv = data['results_df']['Xv'].max()

        # ML коррекции
        ml_correction_mean = 0
        if 'ml_corrections' in data and not data['ml_corrections'].empty:
            ml_correction_mean = data['ml_corrections']['correction'].mean() * 100

        summary_data.append({
            'batch_id': batch_id,
            'final_titer_g_L': round(final_titer, 2),
            'peak_Xv_1e6_per_mL': round(peak_xv, 1),
            'mean_ml_correction_percent': round(ml_correction_mean, 1),
            'mean_r2': round(batch_metrics['_summary']['Mean_R2'], 3),
            'mean_mape': round(batch_metrics['_summary']['Mean_MAPE'], 1)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, "summary.csv")
    try:
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ Сводная таблица сохранена: {summary_csv_path}")
    except Exception as e:
        print(f"⚠️  Ошибка сохранения CSV: {str(e)}")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"✅ Сводная таблица сохранена (стандартная кодировка)")

    # Вывод в консоль
    print(f"\n📊 СВОДНЫЕ РЕЗУЛЬТАТЫ:")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    # Пример использования
    import argparse

    parser = argparse.ArgumentParser(description='Гибридный симулятор CHO')
    parser.add_argument('--batch', type=str, help='Номер партии (01, 02, ...)')
    parser.add_argument('--all', action='store_true', help='Запустить все партии')
    parser.add_argument('--no-ml', action='store_true', help='Без ML коррекции')

    args = parser.parse_args()

    if args.all:
        # Запуск всех партий
        run_all_hybrid_simulations(use_ml_correction=not args.no_ml)
    elif args.batch:
        # Запуск конкретной партии
        csv_path = f"data/raw/batch_CHO{args.batch}.csv"
        meta_path = f"data/meta/batch_CHO{args.batch}.json"
        batch_id = f"CHO{args.batch}"

        run_hybrid_simulation(
            csv_path=csv_path,
            meta_path=meta_path,
            output_path=f"data/hybrid_results/simulation_{batch_id}",
            batch_id=batch_id,
            use_ml_correction=not args.no_ml
        )
    else:
        print("Использование:")
        print("  --all              Запустить все партии")
        print("  --batch <номер>    Запустить конкретную партию")
        print("  --no-ml            Без ML коррекции")
