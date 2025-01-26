import openai
import requests
from bs4 import BeautifulSoup
import numpy as np
import streamlit as st

# Задайте API-ключ OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# URL страницы с курсами
url = "https://karpov.courses"

# Функция для получения эмбеддинга с использованием OpenAI
def get_gpt_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

# Функция для получения курсов с сайта
def get_courses_from_website():
    # Сделаем запрос к странице
    response = requests.get(url)

    # Проверим успешность запроса
    if response.status_code == 200:
        print("Страница успешно загружена!")
    else:
        print("Не удалось загрузить страницу")

    # Парсим HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Ищем все ссылки с классом, который связан с курсами
    courses = soup.find_all('a', class_='t978__innermenu-link')

    # Извлекаем название и URL каждого курса
    valid_courses = []
    for course in courses:
        title = course.find('span', class_='t978__link-inner_left').text.strip()
        link = course['href']
        valid_courses.append((title, link))

    return valid_courses

# Рекомендация курса
def recommend_course(user_query, courses):
    # Получаем эмбеддинг для запроса пользователя
    query_embedding = get_gpt_embedding(user_query).astype('float32').reshape(1, -1)
    
    # Подготовим эмбеддинги для каждого курса
    course_embeddings = [get_gpt_embedding(course[0]) for course in courses]

    # Вычисляем расстояния между запросом и курсами
    distances = [np.linalg.norm(query_embedding - course_embedding) for course_embedding in course_embeddings]

    # Находим курс с наименьшим расстоянием (самый подходящий)
    best_match_idx = np.argmin(distances)
    best_course = courses[best_match_idx]

    return best_course, distances[best_match_idx]

# Создание интерфейса Streamlit
st.title("Рекомендатор курсов")
st.write("Введите свои интересы, и мы подберём для вас подходящий курс!")

user_input = st.text_input("Что вы хотите изучить?", placeholder="Например, машинное обучение, SQL, анализ данных...")

if user_input:
    with st.spinner("Ищем подходящий курс..."):
        # Получаем список курсов с сайта
        courses = get_courses_from_website()

        if courses:
            # Рекомендация курса
            recommended_course, distance = recommend_course(user_input, courses)

            # Вывод результата
            st.success(f"Мы рекомендуем вам курс: **{recommended_course[0]}**")
            st.write(f"Подробнее по ссылке: {recommended_course[1]}")
        else:
            st.error("Не удалось найти курсы. Попробуйте снова позже.")
