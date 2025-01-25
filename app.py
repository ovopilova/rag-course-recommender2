import streamlit as st
import openai
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from typing import List

# Set API keys (make sure to have secrets setup in Streamlit Cloud)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Шаг 1: Получение списка курсов с karpov.courses
def fetch_courses():
    # URL с курсами
    url = "https://karpov.courses"
    
    # Запрос на страницу
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Поиск элементов с курсами (будет зависеть от структуры сайта)
    courses = []
    for course in soup.find_all("div", class_="course-card"):
        name = course.find("h3").text.strip()
        description = course.find("p").text.strip()
        link = course.find("a")["href"]
        courses.append({"name": name, "description": description, "link": link})
    
    return courses

# Шаг 2: Получение эмбеддингов
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response['data'][0]['embedding'])

# Построение базы данных с эмбеддингами
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
st.title("Рекомендатор курсов по Data Science")
st.write("Введите, что вы хотите изучить, и мы найдём лучший курс для вас!")

# Ввод пользователя
user_input = st.text_input("Что вы хотите изучить?", placeholder="Например: нейронные сети, Python, анализ данных...")

if user_input:
    with st.spinner("Ищем подходящий курс..."):
        try:
            # Шаг 3: Построение базы данных
            courses_data = fetch_courses()  # Получаем курсы с сайта
            index, courses_data = build_vector_db(courses_data)  # Строим индекс

            # Шаг 4: Рекомендация курса
            recommended_course = recommend_course(user_input, index, courses_data)

            # Отображение результата
            st.success(f"Мы рекомендуем курс: **{recommended_course['name']}**")
            st.write(f"Описание: {recommended_course['description']}")
            st.write(f"Подробнее: [Перейти к курсу]({recommended_course['link']})")
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
