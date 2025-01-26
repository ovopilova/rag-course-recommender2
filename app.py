import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from transformers import pipeline

# Инициализируем генератор с использованием модели GPT-2 от HuggingFace
generator = pipeline('text-generation', model='gpt2')

# Функция для получения эмбеддинга через GPT-2
def get_gpt2_embedding(text):
    # Генерируем текст продолжения с использованием GPT-2
    result = generator(text, max_length=50)
    generated_text = result[0]['generated_text']
    
    # Возвращаем эмбеддинг (например, можно взять вектор из первой части текста, преобразованного в числовой вектор)
    # Для простоты будем использовать токенизацию и получение эмбеддинга
    tokens = generator.tokenizer.encode(generated_text, return_tensors='pt')
    embedding = generator.model.transformer.wte(tokens)
    
    # Используем среднее значение токенов как эмбеддинг
    return embedding.mean(dim=1).detach().numpy()

# Парсим курсы с сайта Karpov
def get_courses():
    url = "https://karpov.courses"
    response = requests.get(url)

    if response.status_code != 200:
        print("Ошибка при загрузке страницы.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    courses = soup.find_all('a', class_='t978__innermenu-link')

    course_list = []
    for course in courses:
        title = course.find('span', class_='t978__link-inner_left').text.strip()
        link = course['href']
        course_list.append((title, link))

    return course_list

# Функция для поиска курса, который подходит по запросу
def recommend_course(user_input, courses):
    course_embeddings = []
    course_titles = []

    for title, _ in courses:
        course_titles.append(title)
        course_embeddings.append(get_gpt2_embedding(title))  # Эмбеддинги для каждого курса

    query_embedding = get_gpt2_embedding(user_input)  # Эмбеддинг запроса

    similarities = cosine_similarity([query_embedding], course_embeddings)
    recommended_index = similarities.argmax()

    return course_titles[recommended_index], courses[recommended_index][1]

# Streamlit интерфейс
def main():
    st.title("Рекомендатор курсов Karpov.Courses")
    st.write("Введите, что вы хотите изучить, и мы порекомендуем подходящий курс!")

    # Получаем курсы
    courses = get_courses()

    if not courses:
        st.error("Не удалось получить курсы.")
        return

    user_input = st.text_input("Что вы хотите изучить?", "")

    if user_input:
        recommended_course, course_link = recommend_course(user_input, courses)
        st.write(f"Мы рекомендуем вам курс: **{recommended_course}**")
        st.write(f"Вы можете найти его [здесь]({course_link})")

if __name__ == "__main__":
    main()
