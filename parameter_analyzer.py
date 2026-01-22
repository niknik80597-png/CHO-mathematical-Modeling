"""
Анализатор параметров модели CHO для дипломной работы
Анализирует JSON файлы параметров и создает сводную таблицу
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Настройки отображения
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class ParameterAnalyzer:
    """
    Класс для анализа параметров модели
    """

    # Словарь описаний параметров (можно расширять)
    PARAM_DESCRIPTIONS = {
        # ================== КИНЕТИЧЕСКИЕ ПАРАМЕТРЫ ==================
        "mu_max": {
            "name_ru": "Максимальная удельная скорость роста",
            "symbol": "μ_max",
            "unit": "1/ч",
            "description": "Максимальная скорость деления клеток при оптимальных условиях",
            "role": "Определяет скорость экспоненциального роста",
            "typical_range": "0.03-0.08 1/ч",
            "lit_value": "0.05-0.07 1/ч",
            "source": "Bioprocess Engineering Principles, 2nd ed."
        },
        "K_s": {
            "name_ru": "Константа насыщения субстратом (глюкоза)",
            "symbol": "K_s",
            "unit": "г/л",
            "description": "Концентрация глюкозы, при которой скорость роста составляет половину максимальной",
            "role": "Определяет сродство клеток к глюкозе",
            "typical_range": "0.1-1.0 г/л",
            "lit_value": "0.3-0.5 г/л",
            "source": "Monod, 1949"
        },
        "Y_XG": {
            "name_ru": "Выход биомассы по глюкозе",
            "symbol": "Y_X/G",
            "unit": "10⁶ кл·мл/г",
            "description": "Количество биомассы, образующееся на единицу потребленной глюкозы",
            "role": "Определяет эффективность использования субстрата для роста",
            "typical_range": "1.2-2.0",
            "lit_value": "1.5-1.8",
            "source": "Bailey & Ollis, Biochemical Engineering"
        },
        "m_G": {
            "name_ru": "Коэффициент поддерживающего метаболизма",
            "symbol": "m_G",
            "unit": "г/(10⁶ кл·ч)",
            "description": "Скорость потребления глюкозы для поддержания жизнедеятельности клеток",
            "role": "Учитывает энергетические затраты на поддержание функций",
            "typical_range": "0.001-0.01",
            "lit_value": "0.003-0.005",
            "source": "Pirt, 1965"
        },
        "Y_LG": {
            "name_ru": "Выход лактата по глюкозе",
            "symbol": "Y_L/G",
            "unit": "г/г",
            "description": "Количество лактата, образующегося на единицу потребленной глюкозы",
            "role": "Характеризует гликолитическую активность клеток",
            "typical_range": "0.2-0.8",
            "lit_value": "0.4-0.6",
            "source": "Zagari et al., 2013"
        },
        "k_d0": {
            "name_ru": "Базовая скорость гибели клеток",
            "symbol": "k_d0",
            "unit": "1/ч",
            "description": "Скорость гибели клеток в оптимальных условиях",
            "role": "Определяет базовый уровень апоптоза и некроза",
            "typical_range": "0.0005-0.002",
            "lit_value": "0.0007-0.001",
            "source": "Al-Rubeai et al., 1995"
        },
        "Ki_L": {
            "name_ru": "Константа ингибирования лактатом",
            "symbol": "K_iL",
            "unit": "г/л",
            "description": "Концентрация лактата, вызывающая 50% ингибирование роста",
            "role": "Учитывает токсическое действие лактата",
            "typical_range": "5-25 г/л",
            "lit_value": "10-20 г/л",
            "source": "Miller et al., 1988"
        },
        "Ki_N": {
            "name_ru": "Константа ингибирования аммонием",
            "symbol": "K_iN",
            "unit": "г/л",
            "description": "Концентрация аммония, вызывающая 50% ингибирование роста",
            "role": "Учитывает токсическое действие аммония",
            "typical_range": "0.1-1.0 г/л",
            "lit_value": "0.2-0.5 г/л",
            "source": "Ozturk et al., 1992"
        },
        "alpha": {
            "name_ru": "Коэффициент образования аммония",
            "symbol": "α",
            "unit": "г/(10⁶ кл·ч)",
            "description": "Скорость образования аммония на единицу биомассы",
            "role": "Определяет интенсивность образования аммония",
            "typical_range": "0.00005-0.0002",
            "lit_value": "0.00008-0.00015",
            "source": "Glacken et al., 1986"
        },
        "s": {
            "name_ru": "Коэффициент крутизны сигмоиды",
            "symbol": "s",
            "unit": "безразмерный",
            "description": "Определяет крутизну перехода в функции жизнеспособности",
            "role": "Контролирует резкость снижения продуктивности при падении жизнеспособности",
            "typical_range": "5-15",
            "lit_value": "8-12",
            "source": "Sauer et al., 2000"
        },
        "rho": {
            "name_ru": "Плотность питательной среды",
            "symbol": "ρ",
            "unit": "кг/м³",
            "description": "Плотность культуральной среды",
            "role": "Используется в балансе массы",
            "typical_range": "1000-1100",
            "lit_value": "1000-1050",
            "source": "Стандартное значение"
        },
        "Vcrit": {
            "name_ru": "Критическая жизнеспособность",
            "symbol": "V_crit",
            "unit": "безразмерная",
            "description": "Порог жизнеспособности, ниже которого резко возрастает гибель клеток",
            "role": "Определяет переход к фазе снижения продуктивности",
            "typical_range": "0.8-0.95",
            "lit_value": "0.85-0.90",
            "source": "Frampton et al., 2003"
        },
        "qP_max": {
            "name_ru": "Максимальная удельная скорость продукции",
            "symbol": "q_P,max",
            "unit": "г/(10⁶ кл·ч)",
            "description": "Максимальная скорость продукции антитела на клетку",
            "role": "Определяет потенциал продуктивности клеточной линии",
            "typical_range": "0.001-0.005",
            "lit_value": "0.0015-0.003",
            "source": "Wurm, 2004"
        },
        "KP": {
            "name_ru": "Константа полунасыщения для продукции",
            "symbol": "K_P",
            "unit": "г/л",
            "description": "Концентрация глюкозы, при которой скорость продукции составляет половину максимальной",
            "role": "Определяет влияние глюкозы на продуктивность",
            "typical_range": "0.1-0.5",
            "lit_value": "0.2-0.4",
            "source": "Xie & Wang, 1996"
        },
        "beta": {
            "name_ru": "Коэффициент усиления гибели",
            "symbol": "β",
            "unit": "1/ч",
            "description": "Коэффициент, определяющий увеличение скорости гибели при падении жизнеспособности ниже Vcrit",
            "role": "Моделирует ускоренную гибель в конце культивирования",
            "typical_range": "0.01-0.05",
            "lit_value": "0.02-0.04",
            "source": "Fussenegger et al., 1998"
        },
        "Ki_Lac": {
            "name_ru": "Константа ингибирования лактатом продукции",
            "symbol": "K_iLac,P",
            "unit": "г/л",
            "description": "Концентрация лактата, вызывающая 50% ингибирование продукции",
            "role": "Учитывает влияние лактата на продуктивность",
            "typical_range": "3-10 г/л",
            "lit_value": "5-8 г/л",
            "source": "Ozturk & Palsson, 1991"
        },
        "Ki_NH4": {
            "name_ru": "Константа ингибирования аммонием продукции",
            "symbol": "K_iNH4,P",
            "unit": "мМ",
            "description": "Концентрация аммония, вызывающая 50% ингибирование продукции",
            "role": "Учитывает влияние аммония на продуктивность",
            "typical_range": "5-15 мМ",
            "lit_value": "8-12 мМ",
            "source": "Yang et al., 2000"
        },

        # ================== ТЕМПЕРАТУРНЫЕ КОЭФФИЦИЕНТЫ ==================
        "mu_max_factor": {
            "name_ru": "Температурный коэффициент для μ_max",
            "symbol": "f_μ",
            "unit": "безразмерный",
            "description": "Коэффициент изменения максимальной скорости роста при понижении температуры",
            "role": "Учитывает снижение скорости роста при температурном сдвиге",
            "typical_range": "0.5-0.9",
            "lit_value": "0.6-0.8",
            "source": "Trummer et al., 2006"
        },
        "Y_XG_factor": {
            "name_ru": "Температурный коэффициент для Y_XG",
            "symbol": "f_YXG",
            "unit": "безразмерный",
            "description": "Коэффициент изменения выхода биомассы при понижении температуры",
            "role": "Учитывает изменение эффективности использования глюкозы",
            "typical_range": "0.8-1.2",
            "lit_value": "0.9-1.1",
            "source": "Bollati et al., 2011"
        },
        "Y_LG_factor": {
            "name_ru": "Температурный коэффициент для Y_LG",
            "symbol": "f_YLG",
            "unit": "безразмерный",
            "description": "Коэффициент изменения выхода лактата при понижении температуры",
            "role": "Учитывает изменение метаболизма при температурном сдвиге",
            "typical_range": "0.7-1.0",
            "lit_value": "0.8-0.9",
            "source": "Fox et al., 2005"
        },
        "qP_max_factor": {
            "name_ru": "Температурный коэффициент для qP_max",
            "symbol": "f_qP",
            "unit": "безразмерный",
            "description": "Коэффициент изменения максимальной скорости продукции при понижении температуры",
            "role": "Учитывает увеличение специфической продуктивности при температурном сдвиге",
            "typical_range": "1.2-2.0",
            "lit_value": "1.5-1.8",
            "source": "Yoon et al., 2003"
        },
        "k_d0_factor": {
            "name_ru": "Температурный коэффициент для k_d0",
            "symbol": "f_kd",
            "unit": "безразмерный",
            "description": "Коэффициент изменения скорости гибели при понижении температуры",
            "role": "Учитывает снижение скорости гибели при температурном сдвиге",
            "typical_range": "0.5-1.0",
            "lit_value": "0.7-0.9",
            "source": "Moore et al., 1997"
        },

        # ================== НАЧАЛЬНЫЕ УСЛОВИЯ ==================
        "V": {
            "name_ru": "Начальный объём",
            "symbol": "V_0",
            "unit": "л",
            "description": "Объем культуральной среды в начале культивирования",
            "role": "Определяет масштаб процесса",
            "typical_range": "1.0-10.0 л",
            "lit_value": "Зависит от масштаба",
            "source": "Промышленные данные"
        },
        "TCD": {
            "name_ru": "Начальная плотность клеток",
            "symbol": "X_0",
            "unit": "10⁶ кл/мл",
            "description": "Концентрация клеток при инокуляции",
            "role": "Определяет начальную биомассу",
            "typical_range": "0.2-0.5",
            "lit_value": "0.3-0.4",
            "source": "Butler, 2005"
        },
        "Viab": {
            "name_ru": "Начальная жизнеспособность",
            "symbol": "V_0",
            "unit": "безразмерная",
            "description": "Доля жизнеспособных клеток при инокуляции",
            "role": "Определяет качество инокулюма",
            "typical_range": "0.90-0.98",
            "lit_value": "0.95-0.98",
            "source": "Freshney, 2010"
        },
        "G": {
            "name_ru": "Начальная концентрация глюкозы",
            "symbol": "G_0",
            "unit": "г/л",
            "description": "Концентрация глюкозы в начале культивирования",
            "role": "Определяет начальный запас субстрата",
            "typical_range": "3.0-6.0 г/л",
            "lit_value": "4.0-5.0 г/л",
            "source": "Стандартные протоколы"
        },
        "Lac": {
            "name_ru": "Начальная концентрация лактата",
            "symbol": "L_0",
            "unit": "г/л",
            "description": "Концентрация лактата в начале культивирования",
            "role": "Определяет начальный уровень метаболита",
            "typical_range": "0.05-0.3 г/л",
            "lit_value": "0.1-0.2 г/л",
            "source": "Стандартные протоколы"
        },
        "NH4": {
            "name_ru": "Начальная концентрация аммония",
            "symbol": "N_0",
            "unit": "г/л",
            "description": "Концентрация аммония в начале культивирования",
            "role": "Определяет начальный уровень токсичного метаболита",
            "typical_range": "0.01-0.1 г/л",
            "lit_value": "0.02-0.05 г/л",
            "source": "Стандартные протоколы"
        },
        "P": {
            "name_ru": "Начальная концентрация продукта",
            "symbol": "P_0",
            "unit": "г/л",
            "description": "Концентрация антитела в начале культивирования",
            "role": "Обычно равна нулю",
            "typical_range": "0.0 г/л",
            "lit_value": "0.0 г/л",
            "source": "Стандартные протоколы"
        }
    }

    def __init__(self, meta_dir="data/meta"):
        """
        Инициализация анализатора

        Parameters:
        -----------
        meta_dir : str
            Папка с JSON файлами параметров
        """
        self.meta_dir = meta_dir
        self.parameters_data = {}
        self.all_params = set()

    def load_all_parameters(self):
        """
        Загрузка всех параметров из JSON файлов
        """
        print("📂 Загрузка параметров из JSON файлов...")

        # Находим все JSON файлы
        json_files = glob.glob(os.path.join(self.meta_dir, "batch_*.json"))

        if not json_files:
            print(f"❌ Не найдены JSON файлы в папке {self.meta_dir}")
            return False

        print(f"🔍 Найдено файлов: {len(json_files)}")

        for json_file in json_files:
            batch_id = os.path.basename(json_file).replace(".json", "").replace("batch_", "")
            print(f"  📄 Обработка {batch_id}...")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Собираем параметры из разных разделов
                batch_params = {}

                # Кинетические параметры
                if "kinetics_parameters" in data:
                    batch_params.update(data["kinetics_parameters"])

                # Температурные коэффициенты
                if "temperature_coefficients" in data:
                    # Добавляем префикс для ясности
                    temp_coeffs = data["temperature_coefficients"]
                    for key, value in temp_coeffs.items():
                        batch_params[f"temp_{key}"] = value

                # Начальные условия
                if "initial_conditions" in data:
                    initial = data["initial_conditions"]
                    for key, value in initial.items():
                        batch_params[f"init_{key}"] = value

                # Сохраняем параметры партии
                self.parameters_data[batch_id] = batch_params

                # Добавляем в общий список параметров
                self.all_params.update(batch_params.keys())

                print(f"    ✅ Загружено {len(batch_params)} параметров")

            except Exception as e:
                print(f"    ❌ Ошибка загрузки {json_file}: {str(e)}")

        print(f"📊 Всего уникальных параметров: {len(self.all_params)}")
        return True

    def create_summary_table(self):
        """
        Создание сводной таблицы параметров
        """
        print("\n📊 Создание сводной таблицы параметров...")

        summary_data = []

        for param_name in sorted(self.all_params):
            # Собираем значения параметра из всех партий
            values = []
            sources = []

            for batch_id, batch_params in self.parameters_data.items():
                if param_name in batch_params:
                    values.append(batch_params[param_name])
                    sources.append(batch_id)

            if not values:
                continue

            # Основная статистика
            values_array = np.array(values)
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)

            # Определяем тип параметра
            param_type = self._get_param_type(param_name)

            # Получаем описание из справочника
            description_info = self._get_param_description(param_name)

            # Создаем запись для таблицы
            record = {
                "Тип параметра": param_type,
                "Обозначение (JSON)": param_name,
                "Математическое обозначение": description_info.get("symbol", ""),
                "Название параметра": description_info.get("name_ru", ""),
                "Единицы измерения": description_info.get("unit", ""),
                "Минимальное значение": f"{min_val:.4g}",
                "Максимальное значение": f"{max_val:.4g}",
                "Среднее значение (данные)": f"{mean_val:.4g}",
                "Стандартное отклонение": f"{std_val:.4g}",
                "Коэффициент вариации, %": f"{(std_val / mean_val * 100):.1f}" if mean_val != 0 else "N/A",
                "Типичный диапазон (литература)": description_info.get("typical_range", ""),
                "Среднее по литературе": description_info.get("lit_value", ""),
                "Описание параметра": description_info.get("description", ""),
                "Роль в модели": description_info.get("role", ""),
                "Источник (литература)": description_info.get("source", ""),
                "Партии, где используется": ", ".join(sources),
                "Количество партий": len(sources)
            }

            summary_data.append(record)

        # Создаем DataFrame
        df = pd.DataFrame(summary_data)

        # Сортируем по типу параметра
        type_order = ["kinetics", "temperature", "initial", "other"]
        df["Тип параметра"] = pd.Categorical(df["Тип параметра"], categories=type_order, ordered=True)
        df = df.sort_values("Тип параметра")

        print(f"✅ Создана таблица с {len(df)} параметрами")
        return df

    def _get_param_type(self, param_name):
        """
        Определение типа параметра по его названию
        """
        if param_name.startswith("init_"):
            return "initial"
        elif param_name.startswith("temp_"):
            return "temperature"
        elif param_name in ["mu_max", "K_s", "Y_XG", "m_G", "Y_LG", "k_d0", "Ki_L", "Ki_N",
                            "alpha", "s", "rho", "Vcrit", "qP_max", "KP", "beta", "Ki_Lac", "Ki_NH4"]:
            return "kinetics"
        else:
            return "other"

    def _get_param_description(self, param_name):
        """
        Получение описания параметра из справочника
        """
        # Убираем префиксы для поиска в справочнике
        base_name = param_name.replace("init_", "").replace("temp_", "")

        if base_name in self.PARAM_DESCRIPTIONS:
            return self.PARAM_DESCRIPTIONS[base_name]
        else:
            return {
                "name_ru": "Требуется заполнить",
                "symbol": "?",
                "unit": "?",
                "description": "Требуется заполнить",
                "role": "Требуется заполнить",
                "typical_range": "?",
                "lit_value": "?",
                "source": "?"
            }

    def save_results(self, df, output_dir="results/parameters"):
        """
        Сохранение результатов анализа
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Сохраняем в Excel
        excel_path = os.path.join(output_dir, f"parameters_summary_{timestamp}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Все параметры', index=False)

            # Дополнительные листы по типам параметров
            for param_type in df["Тип параметра"].unique():
                df_type = df[df["Тип параметра"] == param_type]
                sheet_name = self._get_sheet_name(param_type)
                df_type.to_excel(writer, sheet_name=sheet_name, index=False)

        # Сохраняем в CSV
        csv_path = os.path.join(output_dir, f"parameters_summary_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Сохраняем в JSON для программистов
        json_path = os.path.join(output_dir, f"parameters_summary_{timestamp}.json")
        df_dict = df.to_dict(orient='records')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(df_dict, f, indent=2, ensure_ascii=False)

        print(f"💾 Результаты сохранены:")
        print(f"  📊 Excel: {excel_path}")
        print(f"  📄 CSV: {csv_path}")
        print(f"  🗃️  JSON: {json_path}")

        return excel_path, csv_path, json_path

    def _get_sheet_name(self, param_type):
        """
        Получение имени листа Excel для типа параметра
        """
        names = {
            "kinetics": "Кинетические",
            "temperature": "Температурные",
            "initial": "Начальные условия",
            "other": "Прочие"
        }
        return names.get(param_type, param_type)

    def create_visualizations(self, df, output_dir="results/parameters/plots"):
        """
        Создание визуализаций параметров
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Распределение параметров по типам
        plt.figure(figsize=(10, 6))
        type_counts = df["Тип параметра"].value_counts()
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        bars = plt.bar(type_counts.index, type_counts.values, color=colors)

        plt.title('Распределение параметров по типам', fontsize=14, fontweight='bold')
        plt.xlabel('Тип параметра')
        plt.ylabel('Количество параметров')
        plt.xticks(rotation=45)

        # Добавляем значения на бары
        for bar, count in zip(bars, type_counts.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_types.png"), dpi=300)

        # 2. Диапазоны важных кинетических параметров
        kinetic_params = df[df["Тип параметра"] == "kinetics"].copy()
        if not kinetic_params.empty:
            # Выбираем топ-10 параметров по коэффициенту вариации
            kinetic_params["CV"] = kinetic_params["Коэффициент вариации, %"].replace("N/A", "0").astype(float)
            top_params = kinetic_params.nlargest(10, "CV")

            plt.figure(figsize=(12, 8))

            for i, (_, row) in enumerate(top_params.iterrows()):
                min_val = float(row["Минимальное значение"])
                max_val = float(row["Максимальное значение"])
                mean_val = float(row["Среднее значение (данные)"])

                plt.plot([min_val, max_val], [i, i], 'k-', linewidth=3, alpha=0.7)
                plt.plot(mean_val, i, 'ro', markersize=8)

                # Добавляем литературное значение, если есть
                lit_range = row["Среднее по литературе"]
                if lit_range and "?" not in lit_range:
                    try:
                        # Пробуем извлечь среднее из диапазона
                        if "-" in lit_range:
                            lit_vals = [float(x.strip()) for x in lit_range.split("-") if
                                        x.strip().replace('.', '').isdigit()]
                            if lit_vals:
                                lit_mean = np.mean(lit_vals)
                                plt.plot(lit_mean, i, 'g*', markersize=10, alpha=0.7)
                    except:
                        pass

            plt.yticks(range(len(top_params)), top_params["Название параметра"])
            plt.xlabel('Значение параметра')
            plt.title('Диапазоны кинетических параметров', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(['Диапазон данных', 'Среднее (данные)', 'Литературное значение'],
                       loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "kinetic_parameters_ranges.png"), dpi=300)

        # 3. Heatmap корреляций между партиями (по общим параметрам)
        self._create_correlation_heatmap(output_dir)

        print(f"📈 Графики сохранены в: {output_dir}")

    def _create_correlation_heatmap(self, output_dir):
        """
        Создание heatmap корреляций между партиями
        """
        # Создаем матрицу параметров (партии × параметры)
        param_matrix = []
        batch_ids = []

        for batch_id, batch_params in self.parameters_data.items():
            row = []
            for param in sorted(self.all_params):
                row.append(batch_params.get(param, np.nan))
            param_matrix.append(row)
            batch_ids.append(batch_id)

        param_matrix = np.array(param_matrix)

        # Вычисляем корреляции между партиями (по общим параметрам)
        n_batches = len(batch_ids)
        corr_matrix = np.zeros((n_batches, n_batches))

        for i in range(n_batches):
            for j in range(n_batches):
                # Берем только те параметры, которые есть в обеих партиях
                mask = ~np.isnan(param_matrix[i]) & ~np.isnan(param_matrix[j])
                if np.sum(mask) > 1:
                    corr = np.corrcoef(param_matrix[i][mask], param_matrix[j][mask])[0, 1]
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = np.nan

        # Визуализация heatmap
        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr_matrix, cmap='RdYlBu', vmin=-1, vmax=1)

        plt.xticks(range(n_batches), batch_ids, rotation=45)
        plt.yticks(range(n_batches), batch_ids)
        plt.title('Корреляция параметров между партиями', fontsize=14, fontweight='bold')
        plt.colorbar(im, label='Коэффициент корреляции')

        # Добавляем значения в ячейки
        for i in range(n_batches):
            for j in range(n_batches):
                if not np.isnan(corr_matrix[i, j]):
                    plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center',
                             color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_correlation_heatmap.png"), dpi=300)

    def generate_latex_table(self, df, output_path="results/parameters/latex_table.tex"):
        """
        Генерация LaTeX таблицы для диплома
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        latex_content = """\\begin{longtable}{|p{2cm}|p{2.5cm}|p{1.5cm}|p{1.5cm}|p{1.5cm}|p{2cm}|p{4cm}|}
\\caption{Параметры математической модели культивирования клеток CHO} \\label{tab:model_parameters} \\\\
\\hline
\\textbf{Параметр} & \\textbf{Обозначение} & \\textbf{Ед. изм.} & \\textbf{Диапазон} & \\textbf{Среднее} & \\textbf{Литература} & \\textbf{Описание} \\\\
\\hline
\\endfirsthead

\\multicolumn{7}{c}{{\\tablename\\ \\thetable{} -- продолжение}} \\\\
\\hline
\\textbf{Параметр} & \\textbf{Обозначение} & \\textbf{Ед. изм.} & \\textbf{Диапазон} & \\textbf{Среднее} & \\textbf{Литература} & \\textbf{Описание} \\\\
\\hline
\\endhead

\\hline
\\multicolumn{7}{r}{{Продолжение на следующей странице}} \\\\
\\endfoot

\\hline
\\endlastfoot
"""

        # Группируем по типу параметра
        for param_type, group in df.groupby("Тип параметра"):
            latex_content += f"\n% ========== {param_type.upper()} ПАРАМЕТРЫ ==========\n"

            for _, row in group.iterrows():
                param_name = row["Название параметра"]
                symbol = row["Математическое обозначение"]
                unit = row["Единицы измерения"]
                param_range = f"{row['Минимальное значение']}-{row['Максимальное значение']}"
                mean_val = row["Среднее значение (данные)"]
                literature = row["Среднее по литературе"]
                description = row["Описание параметра"]

                # Экранируем спецсимволы для LaTeX
                description = description.replace("%", "\\%").replace("_", "\\_")
                symbol = symbol.replace("_", "\\_")

                latex_content += f"{param_name} & ${symbol}$ & {unit} & {param_range} & {mean_val} & {literature} & {description} \\\\\n\\hline\n"

        latex_content += "\\end{longtable}\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)

        print(f"📝 LaTeX таблица сохранена: {output_path}")
        return latex_content

    def run_analysis(self):
        """
        Запуск полного анализа
        """
        print("=" * 80)
        print("📊 АНАЛИЗАТОР ПАРАМЕТРОВ МОДЕЛИ CHO")
        print("=" * 80)

        # 1. Загрузка данных
        if not self.load_all_parameters():
            return

        # 2. Создание таблицы
        df = self.create_summary_table()

        if df.empty:
            print("❌ Не удалось создать таблицу параметров")
            return

        # 3. Сохранение результатов
        excel_path, csv_path, json_path = self.save_results(df)

        # 4. Создание графиков
        self.create_visualizations(df)

        # 5. Генерация LaTeX таблицы
        latex_table = self.generate_latex_table(df)

        # 6. Вывод сводки
        print(f"\n{'=' * 80}")
        print("🎯 СВОДКА АНАЛИЗА:")
        print(f"{'=' * 80}")
        print(f"Всего параметров: {len(df)}")
        print(f"Кинетические: {len(df[df['Тип параметра'] == 'kinetics'])}")
        print(f"Температурные: {len(df[df['Тип параметра'] == 'temperature'])}")
        print(f"Начальные условия: {len(df[df['Тип параметра'] == 'initial'])}")
        print(f"Прочие: {len(df[df['Тип параметра'] == 'other'])}")

        # Показываем топ-5 параметров с наибольшим разбросом
        print(f"\n📈 Топ-5 параметров с наибольшим разбросом:")
        df["Разброс"] = pd.to_numeric(df["Максимальное значение"]) - pd.to_numeric(df["Минимальное значение"])
        top_variable = df.nlargest(5, "Разброс")

        for i, (_, row) in enumerate(top_variable.iterrows(), 1):
            print(f"  {i}. {row['Название параметра']}: {row['Минимальное значение']}-{row['Максимальное значение']} "
                  f"(разброс: {float(row['Разброс']):.4g})")

        print(f"\n✅ Анализ завершен!")
        print(f"📁 Результаты в: results/parameters/")
        print(f"{'=' * 80}")


def main():
    """
    Основная функция
    """
    import argparse

    parser = argparse.ArgumentParser(description='Анализатор параметров модели CHO')
    parser.add_argument('--input', '-i', default='data/meta',
                        help='Папка с JSON файлами параметров')
    parser.add_argument('--output', '-o', default='results/parameters',
                        help='Папка для сохранения результатов')

    args = parser.parse_args()

    # Создаем анализатор
    analyzer = ParameterAnalyzer(meta_dir=args.input)

    # Запускаем анализ
    analyzer.run_analysis()


if __name__ == "__main__":
    main()