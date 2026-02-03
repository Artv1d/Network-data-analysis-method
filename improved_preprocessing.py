import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def extract_features_from_text(text):
    """Извлекает числовые признаки из текста."""
    if pd.isna(text):
        return {}
    
    text = str(text).lower()
    features = {}
    
    # 1. Опыт работы (ищем числа с "лет", "год", "мес")
    experience_patterns = [
        r'(\d+)\s*(лет|года|год|г\.)',
        r'(\d+)\s*(месяц|мес|м\.)',
        r'опыт\s*(\d+)',
        r'стаж\s*(\d+)'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text)
        if matches:
            years = 0
            for match in matches:
                num = int(match[0])
                unit = match[1] if len(match) > 1 else ''
                if any(u in unit for u in ['мес', 'месяц']):
                    years += num / 12
                else:
                    years += num
            features['experience_years'] = years
            break
    
    # 2. Возраст (из колонки "Пол, возраст")
    age_match = re.search(r'(\d+)\s*(лет|год|г\.)', text)
    if age_match:
        features['age'] = int(age_match.group(1))
    
    # 3. Пол
    if 'муж' in text or 'мужск' in text:
        features['gender_male'] = 1
    elif 'жен' in text or 'женск' in text:
        features['gender_female'] = 1
    
    # 4. Образование (уровень)
    education_keywords = {
        'высшее': 3,
        'неоконченное высшее': 2,
        'среднее специальное': 2,
        'среднее': 1,
        'незаконченное высшее': 2,
        'бакалавр': 3,
        'магистр': 4,
        'кандидат': 5,
        'доктор': 6
    }
    
    for keyword, value in education_keywords.items():
        if keyword in text:
            features['education_level'] = value
            break
    
    # 5. Наличие автомобиля
    if 'авто' in text or 'водитель' in text or 'права' in text:
        features['has_car'] = 1
    
    # 6. Город (размер)
    big_cities = ['москва', 'санкт-петербург', 'спб', 'новосибирск', 'екатеринбург', 'нижний новгород']
    medium_cities = ['казань', 'челябинск', 'омск', 'самара', 'ростов', 'уфа', 'красноярск', 'пермь', 'воронеж']
    
    for city in big_cities:
        if city in text:
            features['city_size'] = 3  # Большой
            break
    else:
        for city in medium_cities:
            if city in text:
                features['city_size'] = 2  # Средний
                break
        else:
            features['city_size'] = 1  # Малый
    
    return features

def preprocess_hh_data():
    """Основная функция предобработки - ИСПРАВЛЕННАЯ ВЕРСИЯ."""
    
    print("=" * 60)
    print("УЛУЧШЕННАЯ ОБРАБОТКА ДАННЫХ HH.RU")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n1. ЗАГРУЗКА ДАННЫХ...")
    df = pd.read_csv(r"C:\Users\Kirill_Satyukov\ML_Homework\Network-data-analysis-method\data\hh.csv")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    print(f"   Загружено: {len(df)} строк, {len(df.columns)} колонок")
    
    # 2. Очистка зарплаты и создание маски валидных строк
    print("\n2. ОЧИСТКА ЗАРПЛАТЫ...")
    
    def clean_salary(value):
        try:
            if pd.isna(value):
                return np.nan
            
            value_str = str(value)
            value_str = value_str.replace(' ', '').replace(',', '.')
            value_str = re.sub(r'[^\d\.\-]', '', value_str)
            
            if value_str == '' or value_str == '.':
                return np.nan
            
            salary = float(value_str)
            
            # Разумные границы зарплат
            if salary < 1000 or salary > 1000000:
                return np.nan
            
            return salary
        except:
            return np.nan
    
    df['salary_clean'] = df['ЗП'].apply(clean_salary)
    
    # Создаем маску валидных строк
    valid_mask = df['salary_clean'].notna()
    df_valid = df[valid_mask].copy()
    
    print(f"   Валидных строк: {len(df_valid)} из {len(df)}")
    print(f"   Диапазон зарплат: {df_valid['salary_clean'].min():,.0f} - {df_valid['salary_clean'].max():,.0f} руб.")
    print(f"   Средняя зарплата: {df_valid['salary_clean'].mean():,.0f} руб.")
    
    y = df_valid['salary_clean'].values.astype(np.float32)
    
    # 3. Извлечение признаков только из валидных строк
    print("\n3. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ...")
    
    all_features = []
    
    for idx, row in df_valid.iterrows():
        if idx % 10000 == 0 and idx > 0:
            print(f"   Обработано {idx}/{len(df_valid)} строк...")
        
        features = {}
        
        # Извлекаем признаки из каждой колонки
        for col in ['Пол, возраст', 'Опыт (двойное нажатие для полной версии)', 
                   'Образование и ВУЗ', 'Город', 'Авто']:
            if col in row:
                col_features = extract_features_from_text(row[col])
                features.update(col_features)
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Заполняем пропуски
    default_values = {
        'experience_years': 0,
        'age': 30,
        'education_level': 2,
        'city_size': 2,
        'gender_male': 0,
        'gender_female': 0,
        'has_car': 0
    }
    
    for col, default in default_values.items():
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(default)
        else:
            features_df[col] = default
    
    print(f"   Извлечено базовых признаков: {len(features_df.columns)}")
    
    # 4. One-Hot Encoding для валидных строк
    print("\n4. СОЗДАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ...")
    
    categorical_cols = ['Ищет работу на должность:', 'Город', 'Занятость', 'График',
                       'Последенее/нынешнее место работы', 'Последеняя/нынешняя должность',
                       'Образование и ВУЗ']
    
    categorical_features = pd.DataFrame()
    
    for col in categorical_cols:
        if col in df_valid.columns:
            # Топ-8 самых частых значений
            top_values = df_valid[col].value_counts().head(8).index.tolist()
            
            for value in top_values:
                if pd.notna(value):
                    new_col_name = f"{col[:20]}_{str(value)[:20]}".replace(':', '_').replace('/', '_')
                    new_col_name = re.sub(r'[^\w_]', '', new_col_name)
                    categorical_features[new_col_name] = (df_valid[col] == value).astype(int)
    
    print(f"   Создано категориальных признаков: {len(categorical_features.columns)}")
    
    # 5. Объединение признаков
    print("\n5. ОБЪЕДИНЕНИЕ ПРИЗНАКОВ...")
    
    X_df = pd.concat([features_df, categorical_features], axis=1)
    
    # Проверяем размеры
    print(f"   Размер X: {X_df.shape}")
    print(f"   Размер y: {len(y)}")
    
    if X_df.shape[0] != len(y):
        print(f"   ⚠️ Ошибка размеров! Исправляем...")
        min_size = min(X_df.shape[0], len(y))
        X_df = X_df.iloc[:min_size]
        y = y[:min_size]
        print(f"   Новый размер: {min_size}")
    
    # Удаляем колонки с нулевой дисперсией
    X_df = X_df.loc[:, X_df.std() > 0]
    
    print(f"   Итоговые признаки: {X_df.shape[1]}")
    print(f"   Образцов: {X_df.shape[0]}")
    
    # 6. Сохранение
    print("\n6. СОХРАНЕНИЕ ДАННЫХ...")
    
    output_dir = r"C:\Users\Kirill_Satyukov\ML_Homework\Network-data-analysis-method\results_improved"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    X_array = X_df.values.astype(np.float32)
    y_array = y.astype(np.float32)
    
    np.save(os.path.join(output_dir, 'X_data.npy'), X_array)
    np.save(os.path.join(output_dir, 'y_data.npy'), y_array)
    
    # Сохраняем имена признаков
    feature_names = X_df.columns.tolist()
    np.save(os.path.join(output_dir, 'feature_names.npy'), np.array(feature_names, dtype=object))
    
    # Информационный файл
    with open(os.path.join(output_dir, 'data_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Дата обработки: {pd.Timestamp.now()}\n")
        f.write(f"Образцов: {X_array.shape[0]}\n")
        f.write(f"Признаков: {X_array.shape[1]}\n")
        f.write(f"Зарплаты: {y_array.min():,.0f} - {y_array.max():,.0f} руб.\n")
    
    print(f"\n✅ ДАННЫЕ СОХРАНЕНЫ!")
    print(f"   X: {X_array.shape}")
    print(f"   y: {y_array.shape}")
    
    # 7. Быстрая проверка (опционально, можно закомментировать)
    try:
        print("\n7. БЫСТРАЯ ПРОВЕРКА...")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   R² быстрой проверки: {r2:.4f}")
        
    except Exception as e:
        print(f"   Проверка пропущена: {e}")
    
    return X_array, y_array, feature_names

if __name__ == "__main__":
    preprocess_hh_data()