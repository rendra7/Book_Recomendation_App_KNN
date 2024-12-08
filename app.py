import pickle
import streamlit as st
import numpy as np
import zipfile

st.title("Welcome to the 📚 Book Recommendation Hub 🎯")

# Loading the pre-trained models and datasets
knn_model = pickle.load(open('artifacts/knn_model.pkl', 'rb'))
final_data = pickle.load(open('artifacts/final_data.pkl', 'rb'))
with zipfile.ZipFile('artifacts/book_pivot.pkl.zip', 'r') as zip_ref:
    with zip_ref.open('book_pivot.pkl') as file:
        books_pivot = pickle.load(file)

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

def recommend_books(book_name):
    book_list = []
    book_id = np.where(books_pivot.index == book_name)[0][0]
    distances, suggestions = knn_model.kneighbors(books_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=9)  # Changed to 9 to get 8 recommendations + 1 input book

    poster_urls = fetch_poster(suggestions)

    for i in range(len(suggestions)):
        recommended_books = books_pivot.index[suggestions[i]]
        for book in recommended_books:
            book_list.append(book)

    return book_list, poster_urls

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
