import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from model.simulator import simulation_step, temperature_phase
from model.utils import create_interpolators_from_df
import matplotlib.pyplot as plt
import os


def run_simulation(csv_path: str, meta_path: str, output_path: str = None, batch_id: str = None):
    """
    Основной цикл моделирования fed-batch процесса
    """
    
    # --- ЗАГРУЗКА ДАННЫХ ---
    df_exp = pd.read_csv(csv_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    batch_id = batch_id or meta.get("batch_id", "unknown")
    print(f"🚀 Моделирование партии: {batch_id}")

    # --- ПОДГОТОВКА ДАННЫХ ---
    time_points = df_exp['time_h'].values

    # Интерполяция для feed данных
    feed_glucose_interp = interp1d(
        time_points, df_exp['feed_glucose_gph'].values,
        bounds_error=False, fill_value=0.0
    )
    feed_other_interp = interp1d(
        time_points, df_exp['feed_other_gph'].values,
        bounds_error=False, fill_value=0.0
    )

    # Коррекция аммония (если в ммоль/л)
    if df_exp['ammonium_g_L'].max() > 1.0:
        print(f"    Внимание: аммоний в {batch_id}, вероятно, в ммоль/л.")
        df_exp['ammonium_g_L'] = df_exp['ammonium_g_L'] * 0.018

    # --- НАЧАЛЬНЫЕ УСЛОВИЯ ---
    initial = meta["initial_conditions"]

    # Нормализация аммония
    if initial["NH4"] < 0.05:
        initial["NH4"] = 0.0

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
    
    print(f"    Температурный сдвиг запланирован на: {t_shift} ч")

    # --- ЦИКЛ МОДЕЛИРОВАНИЯ ---
    results = []
    rates_history = []

    # Время моделирования
    total_duration = meta["process_time"]["total_duration_h"]
    simulation_times = np.arange(0, total_duration + dt, dt)
    temp_coeffs = meta["temperature_coefficients"]

    for i, t in enumerate(simulation_times):
        # Получаем входные данные в момент времени t
        F_glc = float(feed_glucose_interp(t))
        F_other = float(feed_other_interp(t))

        # Сохраняем текущее состояние
        results.append(state.copy())

        # Выполняем шаг моделирования
        state, rates = simulation_step(
            state=state,
            inputs=(F_glc, F_other),
            params=params,
            dt=dt,
            t_shift=t_shift,
            temp_coeffs=temp_coeffs
        )

        # Прогресс
        if i % 10 == 0:
            progress = t / total_duration * 100
            print(f"  Прогресс: {progress:.1f}% (t={t:.0f} ч, Xv={state['Xv']:.2f}×10⁶ кл/мл, T_phase={rates.get('T_phase', 0)})")

        # Сохраняем скорости
        rates["time_h"] = t
        rates_history.append(rates)

    # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
    results_df = pd.DataFrame(results)
    rates_df = pd.DataFrame(rates_history)
    results_df["NH4_mM"] = results_df["NH4"] / 0.018
    results_df["batch_id"] = batch_id
    results_df["t_shift"] = t_shift  # Сохраняем время сдвига в DataFrame

    # Сохраняем жизнеспособность из модели
    results_df["viability"] = results_df["Xv"] / results_df["TCD"]

    if output_path:
        # Создаем директорию, если нет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        base_name = f"{output_path}_{batch_id}"
        results_df.to_csv(f"{base_name}_states.csv", index=False)
        rates_df.to_csv(f"{base_name}_rates.csv", index=False)

        # Сохраняем сравнение с экспериментальными данными
        comparison_df = create_comparison_df(results_df, df_exp, batch_id)
        comparison_df.to_csv(f"{base_name}_comparison.csv", index=False)

        print(f"✅ Результаты сохранены в: {base_name}_*.csv")
    else:
        comparison_df = create_comparison_df(results_df, df_exp, batch_id)

    print(f"🎯 Итоги {batch_id}:")
    print(f"   - Конечный титр: {results_df['P'].iloc[-1]:.2f} г/л")
    print(f"   - Пиковая Xv: {results_df['Xv'].max():.2f} ×10⁶ кл/мл")
    print(f"   - Пиковая TCD: {results_df['TCD'].max():.2f} ×10⁶ кл/мл")
    print(f"   - Финальная жизнеспособность: {results_df['viability'].iloc[-1]:.2%}")
    print(f"   - Температурный сдвиг: {t_shift} ч")

    return results_df, rates_df, comparison_df


def create_comparison_df(model_df, exp_df, batch_id):
    """Создание DataFrame для сравнения модели и эксперимента"""
    comparison = pd.DataFrame()

    # Временные точки модели
    model_times = model_df['time_h'].values

    # Интерполируем экспериментальные данные на времена модели
    for col in ['TCD_1e6_per_mL', 'viability_frac', 'glucose_g_L',
                'lactate_g_L', 'titer_g_L']:
        if col in exp_df.columns:
            # Убираем NaN
            exp_times = exp_df['time_h'].values
            exp_values = exp_df[col].values

            # Создаем интерполятор
            valid_idx = ~np.isnan(exp_values)
            if np.sum(valid_idx) > 1:
                interp_func = interp1d(
                    exp_times[valid_idx],
                    exp_values[valid_idx],
                    bounds_error=False,
                    fill_value=np.nan
                )
                comparison[f'exp_{col}'] = interp_func(model_times)
            else:
                comparison[f'exp_{col}'] = np.nan

    # Для аммония - специальная обработка
    if 'ammonium_g_L' in exp_df.columns:
        exp_times = exp_df['time_h'].values
        exp_amm = exp_df['ammonium_g_L'].values
        valid_idx = ~np.isnan(exp_amm)

        if np.sum(valid_idx) > 1:
            interp_func = interp1d(
                exp_times[valid_idx],
                exp_amm[valid_idx],
                bounds_error=False,
                fill_value=np.nan
            )
            comparison['exp_ammonium_g_L'] = interp_func(model_times)
            comparison['exp_ammonium_mM'] = comparison['exp_ammonium_g_L'] / 0.018
        else:
            comparison['exp_ammonium_g_L'] = np.nan
            comparison['exp_ammonium_mM'] = np.nan

    # Добавляем данные модели
    comparison['model_Xv'] = model_df['Xv']
    comparison['model_TCD'] = model_df['TCD']
    comparison['model_G'] = model_df['G']
    comparison['model_Lac'] = model_df['Lac']
    comparison['model_NH4_gL'] = model_df['NH4']
    comparison['model_NH4_mM'] = model_df['NH4_mM']
    comparison['model_P'] = model_df['P']
    comparison['model_viability'] = model_df['Xv'] / model_df['TCD']
    comparison['time_h'] = model_times
    comparison['batch_id'] = batch_id

    return comparison


def plot_single_batch(results_df, rates_df, comparison_df=None, t_shift=None, save_path=None):
    """Визуализация результатов для одной партии"""
    batch_id = results_df.get('batch_id', 'unknown').iloc[0] if 'batch_id' in results_df else 'unknown'
    
    # Получаем время сдвига из results_df, если не передано
    if t_shift is None and 't_shift' in results_df.columns:
        t_shift = results_df['t_shift'].iloc[0]
    
    if t_shift is not None:
        print(f"📊 Температурный сдвиг на {t_shift} ч для партии {batch_id}")
    else:
        print(f"⚠️ Температурный сдвиг не найден для партии {batch_id}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Функция для добавления вертикальной линии на график
    def add_temp_shift_line(ax, t_shift):
        if t_shift is not None and t_shift > 0:
            # Получаем текущие пределы оси Y
            ymin, ymax = ax.get_ylim()
            # Добавляем вертикальную пунктирную линию
            ax.axvline(x=t_shift, color='orange', linestyle='--',
                      linewidth=1.5, alpha=0.7, label=f'Темп. сдвиг ({t_shift} ч)')
            # Восстанавливаем пределы оси Y
            ax.set_ylim(ymin, ymax)
            # Добавляем аннотацию
            ax.text(t_shift, ymax * 0.95, f' {t_shift} ч',
                   color='orange', fontsize=9, verticalalignment='top')
            return True
        return False

    # 1. Биомасса (Xv)
    ax = axes[0, 0]
    ax.plot(results_df['time_h'], results_df['Xv'], 'b-', linewidth=2, label='Модель (Xv)')
    if comparison_df is not None and 'exp_TCD_1e6_per_mL' in comparison_df.columns:
        exp_viab = comparison_df.get('exp_viability_frac', 0.95)
        exp_xv = comparison_df['exp_TCD_1e6_per_mL'] * exp_viab
        ax.plot(comparison_df['time_h'], exp_xv, 'ro',
                markersize=6, alpha=0.7, label='Эксперимент')
    has_shift_line = add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Xv, 10⁶ кл/мл', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Жизнеспособные клетки', fontsize=12)

    # 2. Глюкоза
    ax = axes[0, 1]
    ax.plot(results_df['time_h'], results_df['G'], 'g-', linewidth=2, label='Модель')
    if comparison_df is not None and 'exp_glucose_g_L' in comparison_df.columns:
        ax.plot(comparison_df['time_h'], comparison_df['exp_glucose_g_L'],
                'go', markersize=6, alpha=0.7, label='Эксперимент')
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Глюкоза, г/л', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Глюкоза', fontsize=12)

    # 3. Титр
    ax = axes[0, 2]
    ax.plot(results_df['time_h'], results_df['P'], 'm-', linewidth=2, label='Модель')
    if comparison_df is not None and 'exp_titer_g_L' in comparison_df.columns:
        ax.plot(comparison_df['time_h'], comparison_df['exp_titer_g_L'],
                'mo', markersize=6, alpha=0.7, label='Эксперимент')
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Титр, г/л', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Моноклональные антитела', fontsize=12)

    # 4. Лактат
    ax = axes[1, 0]
    ax.plot(results_df['time_h'], results_df['Lac'], 'r-', linewidth=2, label='Модель')
    if comparison_df is not None and 'exp_lactate_g_L' in comparison_df.columns:
        ax.plot(comparison_df['time_h'], comparison_df['exp_lactate_g_L'],
                'ro', markersize=6, alpha=0.7, label='Эксперимент')
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Лактат, г/л', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Лактат', fontsize=12)

    # 5. Аммоний
    ax = axes[1, 1]
    ax.plot(results_df['time_h'], results_df['NH4_mM'], 'c-', linewidth=2, label='Модель')
    if comparison_df is not None and 'exp_ammonium_mM' in comparison_df.columns:
        ax.plot(comparison_df['time_h'], comparison_df['exp_ammonium_mM'],
                'co', markersize=6, alpha=0.7, label='Эксперимент')
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Аммоний, мМ/л', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Аммоний', fontsize=12)

    # 6. Объём
    ax = axes[1, 2]
    ax.plot(results_df['time_h'], results_df['V'], 'k-', linewidth=2)
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Объём, л', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Объём культуральной жидкости', fontsize=12)

    # 7. Жизнеспособность
    ax = axes[2, 0]
    viability = results_df['Xv'] / results_df['TCD']
    ax.plot(results_df['time_h'], viability, 'y-', linewidth=2)
    if comparison_df is not None and 'exp_viability_frac' in comparison_df.columns:
        ax.plot(comparison_df['time_h'], comparison_df['exp_viability_frac'],
                'yo', markersize=6, alpha=0.7)
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('Жизнеспособность', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Жизнеспособность', fontsize=12)

    # 8. Скорость роста
    ax = axes[2, 1]
    ax.plot(rates_df['time_h'], rates_df['mu'], 'b-', linewidth=2)
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('μ, 1/ч', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Удельная скорость роста', fontsize=12)

    # 9. Скорость продукции
    ax = axes[2, 2]
    ax.plot(rates_df['time_h'], rates_df['qP'], 'm-', linewidth=2)
    add_temp_shift_line(ax, t_shift)
    ax.set_xlabel('Время, ч', fontsize=11)
    ax.set_ylabel('qP, г/(10⁶ кл·ч)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Удельная скорость продукции', fontsize=12)

    # Добавляем общую легенду для температурного сдвига, если она есть
    if has_shift_line:
        fig.legend([plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5)],
                  [f'Температурный сдвиг ({t_shift} ч)'],
                  loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=9)

    plt.suptitle(f'Результаты моделирования: {batch_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 График сохранён: {save_path}")
    plt.show()

    return fig


if __name__ == "__main__":
    # --- ЗАПУСК МОДЕЛИ ---
    
    # Выбери номер партии (01, 02, ..., 09)
    num_batch = "01"  # Измени на нужный номер
    
    print(f"=" * 50)
    print(f"Запуск моделирования для партии CHO{num_batch}")
    print(f"=" * 50)
    
    results_df, rates_df, comparison_df, t_shift = run_simulation(
        csv_path=f"data/raw/batch_CHO{num_batch}.csv",
        meta_path=f"data/meta/batch_CHO{num_batch}.json",
        output_path=f"data/processed/simulation",
        batch_id=f"CHO{num_batch}"
    )
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    plot_single_batch(results_df, rates_df, comparison_df, t_shift=t_shift,
                     save_path=f'simulation_results_CHO{num_batch}.png')

    print("=" * 50)
    print("Моделирование завершено!")
    print(f"Конечный титр: {results_df['P'].iloc[-1]:.3f} г/л")
    print(f"Пиковая Xv: {results_df['Xv'].max():.2f} ×10⁶ кл/мл")
    print(f"Время температурного сдвига: {t_shift} ч")
    print("=" * 50)