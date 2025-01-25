import streamlit as st
import openai
import faiss
import requests
import json
import os
from typing import List
import openai
import numpy as np
import faiss
import streamlit as st
# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Курсы для рекомендации
courses = [
    {"name": "Машинное обучение", "description": "Изучите базовые алгоритмы машинного обучения: регрессия, деревья решений, кластеризация."},
    {"name": "SQL для аналитиков", "description": "Научитесь писать SQL-запросы, работать с базами данных и анализировать данные."},
    {"name": "Нейронные сети", "description": "Разработайте свои первые нейронные сети и изучите их архитектуру."},
    {"name": "Анализ данных в Python", "description": "Научитесь анализировать данные с использованием pandas, numpy и matplotlib."},
    {"name": "Продуктовая аналитика", "description": "Изучите метрики аналитики и научитесь строить отчёты для продукта."}
]

# Получение эмбеддингов от OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response['data'][0]['embedding'])

# Построение базы данных для поиска
def build_vector_db(courses):
    descriptions = [course["description"] for course in courses]
    embeddings = [get_embedding(desc) for desc in descriptions]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return index, courses

# Рекомендация курса
def recommend_course(user_query, index, courses):
    query_embedding = get_embedding(user_query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, 1)
    recommended_course = courses[indices[0][0]]
    return recommended_course

# Интерфейс Streamlit
st.title("Рекомендатор курсов")
st.write("Введите, что вы хотите изучить, и мы найдём лучший курс для вас!")

user_input = st.text_input("Что вы хотите изучить?", placeholder="Например: нейронные сети, SQL, анализ данных...")

if user_input:
    with st.spinner("Ищем подходящий курс..."):
        try:
            # Построение базы данных
            index, courses_data = build_vector_db(courses)

            # Рекомендация курса
            recommended_course = recommend_course(user_input, index, courses_data)

            # Отображение результата
            st.success(f"Мы рекомендуем курс: **{recommended_course['name']}**")
            st.write(f"Описание: {recommended_course['description']}")
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
