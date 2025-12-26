import streamlit as st
import pandas as pd
import joblib

# Загрузка модели
model = joblib.load("models/best_model.pkl")

st.title("Анализ риска для здоровья")

# Ввод данных пользователем
age = st.number_input("Возраст", 18, 100)
bmi = st.number_input("BMI", 10.0, 50.0)
sleep = st.number_input("Часы сна", 0, 12)
stress = st.slider("Уровень стресса", 1, 10)

smoking = st.selectbox("Курение", [0, 1])
alcohol = st.selectbox("Алкоголь", [0, 1])
activity = st.selectbox("Физическая активность", [0, 1])
gender = st.selectbox("Пол", [0, 1])

# Кнопка предсказания
if st.button("Предсказать"):
    data = pd.DataFrame([[age, gender, bmi, smoking, alcohol, activity, sleep, stress]],
        columns=['Age','Gender','BMI','Smoking','Alcohol',
                 'PhysicalActivity','SleepHours','StressLevel'])
    
    prediction = model.predict(data)[0]
    st.success("Высокий риск" if prediction == 1 else "Низкий риск")
