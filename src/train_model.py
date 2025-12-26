import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data, preprocess_data

# Загрузка и обработка данных
df = load_data('data/raw/lifestyle_health_data.csv')
df = preprocess_data(df)

# Разделение на признаки и целевую переменную
X = df.drop('HealthRisk', axis=1)
y = df['HealthRisk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение моделей
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Сохранение лучшей модели
joblib.dump(best_model, 'models/best_model.pkl')
print("Лучшая модель сохранена: models/best_model.pkl")
