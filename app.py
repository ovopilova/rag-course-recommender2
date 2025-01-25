import streamlit as st
import requests
from bs4 import BeautifulSoup

# Шаг 1: Получение списка курсов с karpov.courses
def fetch_courses():
    url = "https://karpov.courses"
    
    # Запрос на страницу
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
    else:
        st.error(f"Ошибка при получении страницы: {response.status_code}")
        return []

    # Парсинг курсов с сайта
    courses = []
    for course in soup.find_all("div", class_="course-card"):
        name = course.find("h3").text.strip() if course.find("h3") else "Неизвестно"
        description = course.find("p").text.strip() if course.find("p") else "Описание отсутствует"
        link = course.find("a")["href"] if course.find("a") else "Ссылка отсутствует"
        
        # Дополнительные параметры
        specialization = course.find("span", class_="specialization").text.strip() if course.find("span", class_="specialization") else "Не указано"
        is_free = "Бесплатный" if course.find("span", class_="free-course") else "Платный"
        has_simulator = "Да" if course.find("span", class_="simulator") else "Нет"
        
        courses.append({
            "name": name,
            "description": description,
            "link": link,
            "specialization": specialization,
            "is_free": is_free,
            "has_simulator": has_simulator
        })
    
    return courses

# Шаг 2: Фильтрация курсов по запросу пользователя
def recommend_course(courses, user_query, filter_free=None, filter_simulator=None):
    recommended_courses = []
    for course in courses:
        if user_query.lower() in course['name'].lower() or user_query.lower() in course['description'].lower():
            # Фильтрация по дополнительным параметрам
            if filter_free and filter_free != course['is_free']:
                continue
            if filter_simulator and filter_simulator != course['has_simulator']:
                continue
            recommended_courses.append(course)
    return recommended_courses

# Интерфейс Streamlit
st.title("Рекомендатор курсов по Data Science")
st.write("Введите, что вы хотите изучить, и мы найдём лучший курс для вас!")

# Ввод пользователя
user_input = st.text_input("Что вы хотите изучить?", placeholder="Например: нейронные сети, Python, анализ данных...")

# Фильтры для курсов
filter_free = st.selectbox("Тип курса", ["Все", "Бесплатный", "Платный"])
filter_simulator = st.selectbox("Симуляторы", ["Все", "Да", "Нет"])

if user_input:
    with st.spinner("Ищем подходящий курс..."):
        try:
            # Шаг 3: Получение курсов с сайта
            courses_data = fetch_courses()  # Получаем курсы с сайта
            if not courses_data:
                st.error("Не удалось загрузить курсы с сайта.")
            else:
                # Фильтрация курсов по запросу и дополнительным фильтрам
                recommended_courses = recommend_course(courses_data, user_input, filter_free, filter_simulator)
                
                if recommended_courses:
                    st.write(f"Мы нашли {len(recommended_courses)} курсов, которые могут вас заинтересовать!")
                    for course in recommended_courses:
                        st.subheader(course['name'])
                        st.write(course['description'])
                        st.write(f"Специализация: {course['specialization']}")
                        st.write(f"Тип курса: {course['is_free']}")
                        st.write(f"Симулятор: {course['has_simulator']}")
                        st.write(f"[Ссылка на курс]({course['link']})")
                else:
                    st.write("К сожалению, подходящих курсов не найдено.")
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
