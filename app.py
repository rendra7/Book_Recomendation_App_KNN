import pickle
import streamlit as st
import numpy as np
import requests

# === Title of the app ===
st.set_page_config(page_title="Book Recommendation App 📚", layout="wide")
st.title("📚 Book Recommendation App")

# === Sidebar for Navigation ===
menu = st.sidebar.radio(
    "Navigate",
    ("Recommendation", "About"),
    index=0  # Default ke "Recommendation"
)

# === Fuction to download file from my Huggingface repository model ===
# === Fuction to download file from Hugging Face ===
@st.cache_data
def download_model(url):
    """Fungsi untuk mengunduh model dari URL Hugging Face dan menyimpannya secara lokal."""
    response = requests.get(url)
    if response.status_code == 200:
        with open('book.pkl', 'wb') as file:
            file.write(response.content)
        return 'book.pkl'
    else:
        st.error(f"Gagal mengunduh model. Status kode: {response.status_code}")
        return None

# URL Hugging Face untuk books_pivot
url_books_pivot = 'https://huggingface.co/Rendra7/recomendation_book/resolve/main/book.pkl'

# Unduh books_pivot dari Hugging Face
books_pivot_path = download_model(url_books_pivot)

# === Muat model dan dataset setelah diunduh ===
if books_pivot_path:
    with open(books_pivot_path, 'rb') as file:
        books_pivot = pickle.load(file)
else:
    st.error("Gagal memuat books_pivot")

# === Load pre-trained models and datasets ===

final_data = pickle.load(open('./data.pkl', 'rb'))
knn_model = pickle.load(open('./model_knn.pkl', 'rb'))


# === Function to fetch posters ===
def fetch_poster(suggestion):
    book_titles = []
    poster_urls = []

    for book_id in suggestion:
        book_title = books_pivot.index[book_id]
        book_titles.append(book_title)

    for title in book_titles[0]: 
        row = final_data[final_data['Book-Title'] == title]
        poster_urls.append(row.iloc[0]['Image-URL-L'])

    return poster_urls

# === Function to recommend books ===
def recommend_books(book_name):
    book_list = []
    book_id = np.where(books_pivot.index == book_name)[0][0]
    distances, suggestions = knn_model.kneighbors(books_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=9)

    poster_urls = fetch_poster(suggestions)

    for i in range(len(suggestions)):
        recommended_books = books_pivot.index[suggestions[i]]
        for book in recommended_books:
            book_list.append(book)

    return book_list, poster_urls


# === Page 1: Recommendation ===
if menu == "Recommendation":
    st.header("📖 Discover Your Next Favorite Book!")
    
    selected_book = st.selectbox(
        "Select a book you like and we will recommend similar ones:",
        books_pivot.index
    )

    if st.button('🔍 Discover Your Next Read'):
        recommended_books, posters = recommend_books(selected_book)
        
        # Tampilkan 8 rekomendasi buku dalam 2 baris, masing-masing 4 kolom
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(posters[0], caption=recommended_books[0], use_container_width=True=True)
        with col2:
            st.image(posters[1], caption=recommended_books[1], use_container_width=True=True)
        with col3:
            st.image(posters[2], caption=recommended_books[2], use_container_width=True=True)
        with col4:
            st.image(posters[3], caption=recommended_books[3], use_container_width=True=True)

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.image(posters[4], caption=recommended_books[4], use_container_width=True=True)
        with col6:
            st.image(posters[5], caption=recommended_books[5], use_container_width=True=True)
        with col7:
            st.image(posters[6], caption=recommended_books[6], use_container_width=True=True)
        with col8:
            st.image(posters[7], caption=recommended_books[7], use_container_width=True=True)


# === Page 2: About ===
elif menu == "About":
    st.header("📘 About This Application")
    st.write("""
    **Book Recommendation App** is a powerful machine learning-based web application 
    that helps users discover books similar to the ones they love.
    This application uses a K-Nearest Neighbors (KNN) model to provide personalized recommendations.
    
    ### Features
    - **Book Recommendations**: Select a book, and get 8 similar book suggestions.
    - **Interactive UI**: Easy-to-use interface with an option to search for your next read.
    - **Machine Learning**: Powered by a KNN model trained on book data.
    
    ### How It Works
    - Select a book from the dropdown list.
    - Click the "Discover Your Next Read" button.
    - Enjoy your personalized book recommendations.
    
    Created by **Rendra Dwi Prasetyo**. This project is open-source and can be accessed [here](https://github.com/rendra7/recomendation_book).
    """)
