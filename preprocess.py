import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the movie dataset
movies_df = pd.read_csv('movies.csv')

# Display the columns in the dataset
print("Available columns:", movies_df.columns)

# Combine features to create a single string for each movie using available columns
def combine_features(row):
    # Combine 'genres', 'keywords', 'overview', and 'tagline' columns to create a combined feature
    return str(row['genres']) + " " + str(row['keywords']) + " " + str(row['overview']) + " " + str(row['tagline'])

# Apply the function to combine the features into a new column
movies_df['combined_features'] = movies_df.apply(combine_features, axis=1)

# Convert the combined features to a matrix of TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the cosine similarity matrix to a file
np.save('cosine_similarity_matrix.npy', cosine_sim)

print("Cosine similarity matrix saved to 'cosine_similarity_matrix.npy'")
