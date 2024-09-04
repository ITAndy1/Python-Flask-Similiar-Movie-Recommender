from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import wikipediaapi  # Import Wikipedia API library

# Load the movie dataset
movies_df = pd.read_csv('movies.csv')

# Load the cosine similarity matrix saved from the preprocess step
cosine_sim = np.load('cosine_similarity_matrix.npy')

# Initialize Flask app
app = Flask(__name__)

# Initialize Wikipedia API with a user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='MovieRecommenderApp/1.0 (https://yourwebsite.com; youremail@example.com)'
)

# Function to get Wikipedia URL
def get_wikipedia_url(movie_title):
    page = wiki_wiki.page(movie_title)
    if page.exists():
        return page.fullurl
    return None

# Define the recommendation function
def get_recommendations(movie_title, cosine_sim, movies_df):
    try:
        idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    except IndexError:
        return [("Movie not found. Please select a different movie.", None)]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = list(movies_df['title'].iloc[movie_indices])
    
    # Get Wikipedia URLs for recommended movies
    recommendations_with_links = [(movie, get_wikipedia_url(movie)) for movie in recommendations]
    
    return recommendations_with_links

# Home route
@app.route('/')
def home():
    return render_template('index.html', movies=list(movies_df['title'].values))

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form['movie']
    recommendations = get_recommendations(selected_movie, cosine_sim, movies_df)
    return render_template('recommend.html', selected_movie=selected_movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
