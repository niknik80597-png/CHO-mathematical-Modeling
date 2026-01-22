
"""
ML тренировка для гибридной модели
Обучение Random Forest для коррекции механистической модели
"""

import numpy as np
import pandas as pd
import json
import os
import glob
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки отображения
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_training_data():
    """
    Загрузка данных для обучения ML модели
    Используем ошибки механистической модели как целевую переменную
    """
    print("📂 Загрузка данных для обучения ML...")

    all_features = []
    all_targets = []
    batch_ids = []

    # Находим все файлы с результатами механистической модели
    result_files = glob.glob("data/processed/simulation_*_comparison.csv")

    if not result_files:
        print("⚠️  Не найдены файлы с результатами моделирования!")
        print("   Запустите сначала механистическую модель")
        return None, None, None

    print(f"🔍 Найдено {len(result_files)} файлов с результатами")

    for file_path in result_files:
        try:
            # Извлекаем batch_id из имени файла
            batch_id = os.path.basename(file_path).replace('simulation_', '').replace('_comparison.csv', '')
            print(f"  📄 Обработка {batch_id}...")

            # Загружаем данные
            df = pd.read_csv(file_path)

            # Удаляем строки с NaN в ключевых колонках
            required_cols = ['time_h', 'model_TCD', 'model_G', 'model_Lac',
                             'model_NH4_gL', 'model_P', 'exp_titer_g_L']

            df_clean = df.dropna(subset=required_cols).copy()

            if len(df_clean) < 5:  # Минимум 5 точек данных
                print(f"  ⚠️  {batch_id}: недостаточно данных ({len(df_clean)} точек)")
                continue

            # Рассчитываем целевую переменную (ошибка предсказания титра)
            df_clean['titer_error'] = df_clean['exp_titer_g_L'] - df_clean['model_P']

            # Создаем признаки (features)
            features = pd.DataFrame()

            # 1. Основные состояния модели
            features['time_h'] = df_clean['time_h']
            features['TCD'] = df_clean['model_TCD']
            features['glucose'] = df_clean['model_G']
            features['lactate'] = df_clean['model_Lac']
            features['ammonium'] = df_clean['model_NH4_gL']
            features['model_titer'] = df_clean['model_P']
            features['exp_titer'] = df_clean['exp_titer_g_L']

            # 2. Производные признаки
            features['time_norm'] = features['time_h'] / features['time_h'].max()
            features['glucose_lactate_ratio'] = features['glucose'] / (features['lactate'] + 0.1)
            features['metabolic_quotient'] = features['lactate'] / (features['glucose'] + 0.1)

            # 3. Временные производные (тренды)
            if len(features) > 1:
                features['titer_growth_rate'] = features['model_titer'].diff().fillna(0)
                features['glucose_change_rate'] = features['glucose'].diff().fillna(0)
                features['lactate_change_rate'] = features['lactate'].diff().fillna(0)

            # 4. Квадраты и взаимодействия
            features['TCD_squared'] = features['TCD'] ** 2
            features['glucose_squared'] = features['glucose'] ** 2
            features['TCD_times_time'] = features['TCD'] * features['time_norm']

            # Целевая переменная
            target = df_clean['titer_error'].values

            # Сохраняем batch_id для каждой точки
            batch_array = np.array([batch_id] * len(features))

            # Добавляем к общим данным
            all_features.append(features)
            all_targets.append(target)
            batch_ids.append(batch_array)

            print(f"  ✅ {batch_id}: {len(features)} точек данных")

        except Exception as e:
            print(f"  ❌ Ошибка при обработке {file_path}: {str(e)}")
            continue

    if not all_features:
        print("❌ Не удалось загрузить данные для обучения!")
        return None, None, None

    # Объединяем все данные
    X = pd.concat(all_features, ignore_index=True)
    y = np.concatenate(all_targets)
    batch_ids_array = np.concatenate(batch_ids)

    print(f"\n📊 ЗАГРУЖЕНО ДАННЫХ:")
    print(f"  Всего точек: {len(X)}")
    print(f"  Количество признаков: {X.shape[1]}")
    print(f"  Диапазон ошибок титра: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Средняя ошибка: {y.mean():.3f}")
    print(f"  Партии: {np.unique(batch_ids_array)}")

    return X, y, batch_ids_array


def prepare_features(X):
    """
    Подготовка и масштабирование признаков
    """
    print("\n🔧 Подготовка признаков...")

    # Сохраняем имена признаков
    feature_names = X.columns.tolist()

    # Создаем копию для обработки
    X_processed = X.copy()

    # Заменяем бесконечности на NaN
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

    # Заполняем пропущенные значения
    for col in X_processed.columns:
        if X_processed[col].isnull().any():
            # Для числовых колонок заполняем медианой
            if X_processed[col].dtype in ['float64', 'int64']:
                X_processed[col].fillna(X_processed[col].median(), inplace=True)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    # Сохраняем scaler для использования при предсказании
    os.makedirs("ml_models", exist_ok=True)
    with open("ml_models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Признаки подготовлены:")
    print(f"  Исходные признаки: {len(feature_names)}")
    print(f"  После обработки: {X_scaled.shape[1]}")

    return X_scaled, feature_names, scaler


def train_random_forest(n_estimators=100, max_depth=10, test_size=0.2):
    """
    Обучение Random Forest модели
    """
    print("\n🌲 ОБУЧЕНИЕ RANDOM FOREST МОДЕЛИ")
    print("=" * 50)

    # Загрузка данных
    X_raw, y, batch_ids = load_training_data()
    if X_raw is None:
        return None, {}

    # Подготовка признаков
    X, feature_names, scaler = prepare_features(X_raw)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(y)), test_size=test_size, random_state=42
    )

    print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"  Обучающая выборка: {len(X_train)} точек")
    print(f"  Тестовая выборка: {len(X_test)} точек")
    print(f"  Размерность признаков: {X_train.shape[1]}")

    # Создание и обучение модели
    print("\n🔄 Обучение Random Forest...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,  # Использовать все ядра процессора
        verbose=1
    )

    model.fit(X_train, y_train)

    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Вычисление метрик
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Кросс-валидация
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)

    # Важность признаков
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n✅ МОДЕЛЬ ОБУЧЕНА!")

    # Сохранение модели
    os.makedirs("ml_models", exist_ok=True)
    model_path = "ml_models/random_forest_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"💾 Модель сохранена: {model_path}")

    # Визуализация важности признаков
    plot_feature_importance(sorted_importance[:15])

    # Визуализация предсказаний
    plot_predictions(y_test, y_test_pred, batch_ids[idx_test])

    # Сбор метрик
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'model_params': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'test_size': test_size
        },
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_samples': len(X),
        'n_features': X.shape[1]
    }

    # Сохранение метрик
    with open("ml_models/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


def plot_feature_importance(importance_list):
    """Визуализация важности признаков"""
    features = [item[0] for item in importance_list]
    importance = [item[1] for item in importance_list]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(features)), importance, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Важность признака')
    plt.title('Важность признаков в Random Forest')
    plt.gca().invert_yaxis()

    # Добавляем значения на бары
    for i, (bar, val) in enumerate(zip(bars, importance)):
        plt.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    os.makedirs("ml_models/plots", exist_ok=True)
    plt.savefig("ml_models/plots/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(y_true, y_pred, batch_ids):
    """Визуализация предсказаний"""
    plt.figure(figsize=(12, 5))

    # График 1: Предсказания vs Фактические значения
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, c='steelblue', edgecolor='k', linewidth=0.5)

    # Линия идеального предсказания
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Идеальное предсказание')

    plt.xlabel('Фактическая ошибка')
    plt.ylabel('Предсказанная ошибка')
    plt.title('Предсказания Random Forest')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 2: Распределение ошибок
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Ошибка предсказания')
    plt.ylabel('Частота')
    plt.title(f'Распределение ошибок\nСреднее: {errors.mean():.4f}, STD: {errors.std():.4f}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ml_models/plots/predictions.png", dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_ml_model(model=None):
    """
    Оценка ML модели на всех данных
    """
    print("\n📈 ОЦЕНКА ML МОДЕЛИ")

    if model is None:
        # Загружаем сохраненную модель
        model_path = "ml_models/random_forest_model.pkl"
        if not os.path.exists(model_path):
            print("❌ Модель не найдена!")
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Загрузка данных
    X_raw, y, batch_ids = load_training_data()
    if X_raw is None:
        return None

    # Подготовка признаков (загружаем сохраненный scaler)
    scaler_path = "ml_models/scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X_raw)
    else:
        X, _, _ = prepare_features(X_raw)

    # Предсказания на всех данных
    y_pred = model.predict(X)

    # Вычисление метрик
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Оценка по партиям
    batch_metrics = {}
    unique_batches = np.unique(batch_ids)

    for batch in unique_batches:
        mask = batch_ids == batch
        y_batch = y[mask]
        y_pred_batch = y_pred[mask]

        if len(y_batch) > 1:  # Нужно хотя бы 2 точки для R²
            batch_r2 = r2_score(y_batch, y_pred_batch)
            batch_mae = mean_absolute_error(y_batch, y_pred_batch)
            batch_metrics[batch] = {
                'r2': batch_r2,
                'mae': batch_mae,
                'n_points': len(y_batch)
            }

    results = {
        'overall_r2': r2,
        'overall_mae': mae,
        'overall_rmse': rmse,
        'batch_metrics': batch_metrics,
        'mean_error': np.mean(np.abs(y_pred - y)),
        'std_error': np.std(y_pred - y),
        'n_samples': len(y)
    }

    print(f"\n📊 РЕЗУЛЬТАТЫ ОЦЕНКИ:")
    print(f"  Общий R²: {r2:.3f}")
    print(f"  Общая MAE: {mae:.3f}")
    print(f"  Общая RMSE: {rmse:.3f}")
    print(f"  Средняя абсолютная ошибка: {results['mean_error']:.4f}")

    print(f"\n📈 ПО ПАРТИЯМ:")
    for batch, metrics in batch_metrics.items():
        print(f"  {batch}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f} ({metrics['n_points']} точек)")

    # Визуализация результатов по партиям
    plot_batch_results(results, batch_ids, y, y_pred)

    return results


def plot_batch_results(results, batch_ids, y_true, y_pred):
    """Визуализация результатов по партиям"""
    unique_batches = np.unique(batch_ids)
    n_batches = len(unique_batches)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: R² по партиям
    ax = axes[0, 0]
    batch_r2 = [results['batch_metrics'].get(batch, {}).get('r2', 0) for batch in unique_batches]
    bars = ax.bar(range(n_batches), batch_r2, color='steelblue')
    ax.set_xticks(range(n_batches))
    ax.set_xticklabels(unique_batches, rotation=45)
    ax.set_ylabel('R²')
    ax.set_title('Качество предсказания по партиям (R²)')
    ax.grid(True, alpha=0.3)

    # График 2: Ошибки по партиям
    ax = axes[0, 1]
    batch_errors = []
    for batch in unique_batches:
        mask = batch_ids == batch
        if np.any(mask):
            error = np.mean(np.abs(y_pred[mask] - y_true[mask]))
            batch_errors.append(error)

    bars = ax.bar(range(n_batches), batch_errors, color='coral')
    ax.set_xticks(range(n_batches))
    ax.set_xticklabels(unique_batches, rotation=45)
    ax.set_ylabel('Средняя абсолютная ошибка')
    ax.set_title('Ошибки предсказания по партиям')
    ax.grid(True, alpha=0.3)

    # График 3: Распределение ошибок
    ax = axes[1, 0]
    errors = y_pred - y_true
    ax.hist(errors, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Ошибка предсказания')
    ax.set_ylabel('Частота')
    ax.set_title(f'Распределение ошибок\nСреднее: {errors.mean():.4f}, STD: {errors.std():.4f}')
    ax.grid(True, alpha=0.3)

    # График 4: Предсказания vs Фактические значения
    ax = axes[1, 1]
    scatter = ax.scatter(y_true, y_pred, c=range(len(y_true)),
                         cmap='viridis', alpha=0.6, edgecolor='k', linewidth=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Идеальное предсказание')

    ax.set_xlabel('Фактическая ошибка')
    ax.set_ylabel('Предсказанная ошибка')
    ax.set_title('Предсказания vs Фактические значения')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Порядковый номер точки')

    plt.suptitle('Оценка Random Forest модели по партиям', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("ml_models/plots/batch_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Пример использования
    print("🤖 ML Тренер для гибридной модели CHO")
    model, metrics = train_random_forest()

    if model:
        print(f"\n✅ Обучение завершено успешно!")
        print(f"📊 Тестовая точность (R²): {metrics['test_r2']:.3f}")
    else:
        print("❌ Обучение не удалось")
