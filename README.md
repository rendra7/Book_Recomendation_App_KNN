# 📚 **Book Recommendation App**  
**Created By : Rendra Dwi Prasetyo - 2602199960**
**Live Demo**: [Explore the Book Recommendation App](https://recomendationbook-5nisgqenzcheciarcufgwh.streamlit.app/)  

---

## 🚀 **Project Overview**  
This project focuses on building a **Machine Learning-based Book Recommendation System**. At its core is a **K-Nearest Neighbors (KNN) model** that recommends books similar to the user's selected book. The model analyzes features from a large dataset and uses **distance-based similarity metrics** to identify the most relevant books.  

The project highlights **Data Science skills** such as machine learning, data preprocessing, and model deployment. The app is deployed using **Streamlit**, providing users with an interactive experience to discover their next favorite read.  

---

## 🧠 **Key Features**  
- **Personalized Recommendations**: Uses a KNN-based model to recommend books similar to the user's selection.  
- **Interactive User Interface**: Built with **Streamlit**, enabling users to explore book recommendations.  
- **Efficient Data Handling**: Processes and stores large book datasets efficiently for smooth performance.  

---

## 📂 **Project Structure**  
📦 recomendation_book
┣ 📜 app.py # Main Streamlit app file
┣ 📜 model_knn.pkl # Pre-trained KNN model
┣ 📜 book.pkl # Book feature data
┣ 📜 data.pkl # Additional metadata for recommendations
┣ 📜 .gitattributes # Git LFS tracking for large files
┗ 📜 README.md # This README file

---

## ⚙️ **How It Works**  
1. **Data Loading**: The app loads pre-trained models and datasets (`model_knn.pkl`, `book.pkl`, `data.pkl`).  
2. **User Input**: Users select a book title from the dropdown.  
3. **Recommendation**: The KNN model identifies the closest books based on the selected title.  
4. **Results Display**: Recommended books are displayed with their cover images.  

---

## 📘 **Technologies Used**  
- **Machine Learning**: K-Nearest Neighbors (KNN)  
- **Data Handling**: Pandas, Numpy, and Pickle for dataset processing  
- **Web App**: Streamlit for an interactive user interface  
- **Version Control**: Git and GitHub with Git LFS for large files  

---

## ***Thank You**
