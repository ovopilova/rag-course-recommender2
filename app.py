import os
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
import openai
import streamlit as st
from typing import List, Tuple

def fetch_courses() -> List[str]:
    """
    Fetch course titles from karpov.courses.
    Returns a list of course titles.
    """
    base_url = "https://karpov.courses"
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Failed to fetch courses from karpov.courses.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    course_titles = []

    # Найти все ссылки на курсы
    for course in soup.find_all('a', class_='t978__innermenu-link'):
        try:
            title = course.find('span', class_='t978__link-inner_left').text.strip()
            course_titles.append(title)
        except Exception as e:
            print(f"Error fetching course title: {e}")

    return list(set(course_titles))  # Удалить дубликаты

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Get OpenAI embedding for a given text.
    """
    response = openai.Embedding.create(input=text, model=model)
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

def build_vector_db(course_titles: List[str]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Build a FAISS vector database from course titles.
    Returns the FAISS index and corresponding course titles.
    """
    embeddings = [get_embedding(title) for title in course_titles]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, course_titles

def recommend_course(user_input: str, index: faiss.IndexFlatL2, course_titles: List[str], model: str = "text-embedding-ada-002") -> str:
    """
    Recommend a course based on user input.
    """
    user_embedding = get_embedding(user_input, model=model)
    _, indices = index.search(np.array([user_embedding]), k=1)
    return course_titles[indices[0][0]]

# STREAMLIT APPLICATION
st.title("Course Recommender")

# Get API Key from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Fetch courses
st.write("Fetching courses from karpov.courses...")
course_titles = fetch_courses()

if not course_titles:
    st.error("No courses found. Please check the parsing logic.")
else:
    st.success(f"Fetched {len(course_titles)} courses.")

    # Build vector database
    st.write("Building vector database...")
    index, course_titles = build_vector_db(course_titles)

    # User interaction
    st.write("Let's find the best course for you!")

    if "dialog_state" not in st.session_state:
        st.session_state.dialog_state = "initial"

    if st.session_state.dialog_state == "initial":
        st.write("Hi! What would you like to learn today?")
        user_input = st.text_input("Your answer:")

        if user_input:
            st.session_state.user_input = user_input
            st.session_state.dialog_state = "recommendation"

    elif st.session_state.dialog_state == "recommendation":
        st.write("Thank you! Let me find the best course for you...")
        recommended_course = recommend_course(st.session_state.user_input, index, course_titles)
        st.success(f"Based on what you've told me, I recommend: {recommended_course}")
