🍴 Recipe Recommendation System

A machine learning and NLP-powered web application built with Streamlit.  
It helps users discover recipes based on the ingredients they already have at home, reducing food waste and making cooking easier.

How it works
- Uses TF-IDF Vectorization to represent recipe ingredients as feature vectors
- A K-Nearest Neighbors (KNN) model finds the most similar recipes to the user’s entered ingredients
- Recipes are displayed as interactive cards with images, ratings, calories, protein content, and preparation steps

Features
- Enter multiple ingredients (comma-separated) to get instant recipe recommendations
- View recipes with image, nutrition info, ratings, and procedure
- Expandable "View Procedure" for cooking steps
- Beautiful card-based UI in Streamlit
- Intelligent ingredient matching using TF-IDF + KNN

Tech Stack
- Frontend & UI: Streamlit
- Data Handling: Pandas, NumPy
- Machine Learning: Scikit-learn (TF-IDF, KNN)
- Dataset: Recipe dataset (recipe_final.csv)

How to Run
1. Clone this repo:
   git clone https://github.com/neharakeshpatil/recipe-recommender-system.git
   cd recipe-recommender-system

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app.py

Example
Input:
chicken, garlic, onion

Output (Recommended Recipes):
- Butter Chicken 🍛
- Garlic Chicken Stir Fry 🍲
- Chicken Curry 🥘

Each recommendation shows: ingredients, nutrition, reviews, and procedure.

Future Improvements
- Add personalized recommendations based on dietary preferences
- Support for multiple cuisines (Indian, Italian, Chinese, etc.)
- Integration with image recognition for ingredient detection
