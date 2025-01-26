import openai
import numpy as np
import faiss
import streamlit as st
import time

# Укажите API-ключ OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Список курсов
courses = [
    {"name": "Машинное обучение", "description": "Изучите алгоритмы машинного обучения, такие как регрессия, деревья решений и кластеризация."},
    {"name": "SQL для аналитиков", "description": "Освойте SQL, чтобы работать с базами данных и анализировать данные."},
    {"name": "Нейронные сети", "description": "Изучите основы нейронных сетей и разработайте свои первые модели на Python."},
    {"name": "Анализ данных в Python", "description": "Научитесь анализировать данные с помощью pandas, numpy и matplotlib."},
    {"name": "Продуктовая аналитика", "description": "Изучите основные метрики продуктовой аналитики и научитесь строить отчёты."}
]

# Получение эмбеддингов с помощью OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return np.array(response['data'][0]['embedding'])

# Создание векторной базы данных
def build_vector_db(courses):
    descriptions = [course["description"] for course in courses]
    embeddings = [get_embedding(desc) for desc in descriptions]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

# Рекомендация курса
def recommend_course(user_query, index, courses):
    query_embedding = get_embedding(user_query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, 1)
    recommended_course = courses[indices[0][0]]
    return recommended_course, distances[0][0]

# Создание интерфейса Streamlit
st.title("Рекомендатор курсов")
st.write("Введите свои интересы, и мы подберём для вас подходящий курс!")

user_input = st.text_input("Что вы хотите изучить?", placeholder="Например, машинное обучение, SQL, анализ данных...")

if user_input:
    with st.spinner("Ищем подходящий курс..."):
        # Построение векторной базы
        index, _ = build_vector_db(courses)

        # Рекомендация курса
        recommended_course, distance = recommend_course(user_input, index, courses)

        # Вывод результата
        st.success(f"Мы рекомендуем вам курс: **{recommended_course['name']}**")
        st.write(f"Описание: {recommended_course['description']}")
