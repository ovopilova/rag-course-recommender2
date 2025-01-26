import os
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
import openai
import re
import streamlit as st
from typing import List, Tuple


def fetch_courses() -> List[Tuple[str, str]]:
    """
    Fetch course titles and descriptions from karpov.courses.
    Returns a list of (title, description) tuples.
    """
    base_url = "https://karpov.courses"
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Failed to fetch courses from karpov.courses.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    courses = []

    # Стоп-слова для удаления бесполезных блоков
    stop_words = [
        "оплат", "налоговый вычет", "рассрочк", "гарантия возврата",
        "платеж", "платёж", "скидк", "процент", "работодатель", "карьерный курс",
        "инфраструктур", "финальный проект", "партнер", "старт потока", "бесплатн", 'деньги', 'подробнее', 'вернём', 'лиценз', 'консульт'
    ]

    # Найти все ссылки на курсы
    for course in soup.find_all('a', class_='t978__innermenu-link'):
        try:
            title = course.find('span', class_='t978__link-inner_left').text.strip()
            link = course['href']
            # Зайти на страницу курса и получить подробное описание
            course_response = requests.get(link)
            if course_response.status_code == 200:
                course_soup = BeautifulSoup(course_response.text, 'html.parser')
                infos = course_soup.find_all('div', class_='tn-atom')

                # Оставить только полезные текстовые блоки
                filtered_text = []
                for element in infos:
                    text = element.get_text(strip=True).replace("\xa0", " ")
                    if not any(stop_word in text.lower() for stop_word in stop_words):
                        filtered_text.append(text)

                description = ' '.join(filtered_text)
                courses.append((title, description))
        except Exception as e:
            print(f"Error fetching details for {title}: {e}")

    return list(set(courses))  # Удалить дубликаты


import time

def get_embedding(text: str, model: str = "text-embedding-ada-002", max_retries: int = 5) -> np.ndarray:
    """
    Get OpenAI embedding for a given text with retries for RateLimitError.
    """
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(input=text, model=model)
            return np.array(response["data"][0]["embedding"], dtype=np.float32)
        except openai.error.RateLimitError:
            wait_time = (2 ** attempt) + (0.1 * np.random.random())  # Экспоненциальная задержка с джиттером
            print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Прерываем цикл при других ошибках
    raise Exception("Failed to get embedding after multiple retries.")

def build_vector_db(courses: List[Tuple[str, str]]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Build a FAISS vector database from course descriptions.
    Returns the FAISS index and corresponding course titles.
    """
    descriptions = [desc for _, desc in courses]
    embeddings = [get_embedding(desc) for desc in descriptions]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, [title for title, _ in courses]

# 3. RECOMMEND A COURSE
def recommend_course(user_input: str, index: faiss.IndexFlatL2, courses: List[str], model: str = "text-embedding-ada-002") -> str:
    """
    Recommend a course based on user input.
    """
    user_embedding = get_embedding(user_input, model=model)
    _, indices = index.search(np.array([user_embedding]), k=1)
    return courses[indices[0][0]]

# 4. STREAMLIT APPLICATION
st.title("Course Recommender")

# Get API Key from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Fetch courses
st.write("Fetching courses from karpov.courses...")
courses = fetch_courses()

if not courses:
    st.error("No courses found. Please check the parsing logic.")
else:
    st.success(f"Fetched {len(courses)} courses.")

    # Build vector database
    st.write("Building vector database...")
    index, course_titles = build_vector_db(courses)

    # User interaction
    st.write("Let's find the best course for you!")

    if "dialog_state" not in st.session_state:
        st.session_state.dialog_state = "initial"

    if st.session_state.dialog_state == "initial":
        st.write("Hi! What would you like to learn today?")
        user_input = st.text_input("Your answer:")

        if user_input:
            st.session_state.user_input = user_input
            st.session_state.dialog_state = "follow_up"

    elif st.session_state.dialog_state == "follow_up":
        st.write(f"Got it! You're interested in {st.session_state.user_input}. Could you tell me a bit more about your goals?")
        user_goal = st.text_input("Your goal:")

        if user_goal:
            st.session_state.user_goal = user_goal
            st.session_state.dialog_state = "recommendation"

    elif st.session_state.dialog_state == "recommendation":
        st.write("Thank you for sharing! Let me find the best course for you...")
        recommended_course = recommend_course(st.session_state.user_input, index, course_titles)
        st.success(f"Based on what you've told me, I recommend: {recommended_course}")
