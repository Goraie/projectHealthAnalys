import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """Загрузка данных"""
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """Обработка данных: пропуски, кодирование, масштабирование"""
    # Пропуски
    num_cols = ['Age', 'BMI', 'SleepHours', 'StressLevel']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = ['Gender', 'Smoking', 'Alcohol', 'PhysicalActivity']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Кодирование категориальных признаков
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Масштабирование числовых признаков
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df
