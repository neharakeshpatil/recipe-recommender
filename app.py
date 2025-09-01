import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load and clean the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('recipe_final.csv', encoding='ISO-8859-1')

    # Remove unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 14'], errors='ignore')

    # Parse the ingredients_list column into proper lists
    df['ingredients_list'] = df['ingredients_list'].apply(
        lambda x: [ing.strip().lower() for ing in x.strip('[]').replace("'", "").split(',')] 
        if isinstance(x, str) else []
    )

    return df

# Prepare TF-IDF vectorizer and KNN model
@st.cache_data
def prepare_knn(recipe_df):
    # Convert ingredients list into a single string for each recipe
    recipe_df['ingredients_str'] = recipe_df['ingredients_list'].apply(lambda x: ' '.join(x))

    # Vectorize the ingredients using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_ingredients = vectorizer.fit_transform(recipe_df['ingredients_str'])

    # Train the KNN model using the ingredient vectors
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(X_ingredients)

    return knn, vectorizer

# Recommendation function using KNN and TF-IDF
def recommend_recipes(input_ingredients, knn, vectorizer, recipe_df):
    # Convert input ingredients to a string and vectorize
    input_ingredients_str = ' '.join([ing.strip().lower() for ing in input_ingredients.split(',')])
    input_ingredients_vec = vectorizer.transform([input_ingredients_str])

    # Find the nearest neighbors using KNN
    distances, indices = knn.kneighbors(input_ingredients_vec)

    # Get the recommended recipes
    recommendations = recipe_df.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'image_url', 'aver_rate', 'review_nums', 'calories', 'protein', 'procedure']]

# Streamlit app
def main():
    st.set_page_config(page_title="Recipe Recommendation System", layout="wide")
    st.markdown("""
        <style>
            .title {
                text-align: center;
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 20px;
                color: #333;
            }
        </style>
        <div class="title">🍴 Recipe Recommendation System 🍴</div>
    """, unsafe_allow_html=True)
    st.write("Enter ingredients below to discover recipes that match your ingredients.")

    # Load the dataset
    recipe_df = load_data()

    # Prepare the KNN model based on ingredients
    knn, vectorizer = prepare_knn(recipe_df)

    # User input for ingredients
    ingredients = st.text_input("Enter ingredients (comma-separated):", placeholder="e.g., chicken, garlic, onion")

    if ingredients:
        # Get recommendations using KNN
        recommendations = recommend_recipes(ingredients, knn, vectorizer, recipe_df)
        
        if not recommendations.empty:
            st.subheader("Recommended Recipes")
            # Display recipes in cards
            cols = st.columns(3)  # 3 cards per row
            row_cards = []  # To keep track of card heights for the current row

            for i, (index, row) in enumerate(recommendations.iterrows()):
                # Generate card content
                card_content = f"""
                    <div style="
                        border: 1px solid #ddd; 
                        border-radius: 8px; 
                        padding: 15px; 
                        background-color: #fff; 
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        display: flex; 
                        flex-direction: column; 
                        justify-content: space-between; 
                        height: 100%;
                    ">
                        <img src="{row['image_url'] if pd.notna(row['image_url']) else 'https://via.placeholder.com/300'}" 
                             alt="{row['recipe_name']}" 
                             style="border-radius: 8px; height: 180px; object-fit: cover; width: 100%;">
                        <h3 style="margin-top: 10px;">{row['recipe_name']}</h3>
                        <p style="margin: 5px 0; color: #555;">
                            <strong>Rating:</strong> {row['aver_rate']} ⭐ ({row['review_nums']} reviews)<br>
                            <strong>Calories:</strong> {row['calories']} kcal<br>
                            <strong>Protein:</strong> {row['protein']}g<br>
                            <strong>Ingredients:</strong> {', '.join(row['ingredients_list'])}
                        </p>
                        <details style="margin-top: 10px;">
                            <summary style="cursor: pointer; color: #007BFF;">View Procedure</summary>
                            <p style="margin-top: 5px; color: #666;">{row['procedure'] if pd.notna(row['procedure']) else "Procedure not available."}</p>
                        </details>
                    </div>
                """
                row_cards.append(card_content)

                # If it's the last column in the row or the last recipe, display the row
                if (i + 1) % 3 == 0 or (i + 1) == len(recommendations):
                    # Find the maximum card height (ensures uniformity in the row)
                    row_heights = [len(content) for content in row_cards]
                    max_height = max(row_heights) + 100  # Add padding for longer content
                    row_style = f"min-height: {max_height}px;"

                    # Render all cards in the row
                    cols = st.columns(3)  # Reset columns for the row
                    for col, card in zip(cols, row_cards):
                        with col:
                            st.markdown(card, unsafe_allow_html=True)

                    row_cards = []  # Reset for the next row
        else:
            st.warning("No matching recipes found. Try different ingredients!")

if __name__ == "__main__":
    main()
