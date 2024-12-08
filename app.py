import streamlit as st
import pickle
import numpy as np

# === Set konfigurasi halaman ===
st.set_page_config(page_title="Book Recommendation App 📚", layout="wide")

# === Custom CSS untuk mengubah warna background menu ===
st.markdown("""
    <style>
    .css-1aumxhk {  /* Untuk sidebar */
        background-color: #4B6587 !important;
    }
    .css-17eq0hr {  /* Untuk teks judul */
        color: #F5F5F5 !important;
    }
    .css-qbe2hs {  /* Untuk teks sidebar */
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# === Title aplikasi ===
st.title("📚 Book Recommendation App")

# === Load model dan data ===
knn_model = pickle.load(open('./model_kkn.pkl', 'rb'))
final_data = pickle.load(open('./data.pkl', 'rb'))
books_pivot = pickle.load(open('./book.pkl', 'rb'))

# === Fungsi untuk mengambil poster buku ===
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

# === Fungsi untuk merekomendasikan buku ===
def recommend_books(book_name):
    book_list = []
    book_id = np.where(books_pivot.index == book_name)[0][0]
    distances, suggestions = knn_model.kneighbors(books_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=9)  # 8 rekomendasi + 1 buku input

    poster_urls = fetch_poster(suggestions)

    for i in range(len(suggestions)):
        recommended_books = books_pivot.index[suggestions[i]]
        for book in recommended_books:
            book_list.append(book)

    return book_list, poster_urls

# === Menu navigasi sidebar ===
menu = st.sidebar.selectbox("Menu", ["Recommendation", "About"])

if menu == "Recommendation":
    st.header("📖 Discover Your Next Favorite Book!")
    
    selected_book = st.selectbox(
        "Select a book you like and we will recommend similar ones:",
        books_pivot.index
    )

    if st.button('Discover Your Next Read 📖'):
        recommended_books, posters = recommend_books(selected_book)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(posters[0], caption=recommended_books[0], use_column_width=True)
        with col2:
            st.image(posters[1], caption=recommended_books[1], use_column_width=True)
        with col3:
            st.image(posters[2], caption=recommended_books[2], use_column_width=True)
        with col4:
            st.image(posters[3], caption=recommended_books[3], use_column_width=True)

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.image(posters[4], caption=recommended_books[4], use_column_width=True)
        with col6:
            st.image(posters[5], caption=recommended_books[5], use_column_width=True)
        with col7:
            st.image(posters[6], caption=recommended_books[6], use_column_width=True)
        with col8:
            st.image(posters[7], caption=recommended_books[7], use_column_width=True)


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
