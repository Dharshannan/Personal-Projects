# Movie recommendation through content based filtering
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('movie_dataset.csv')
features = ['genres', 'keywords', 'cast', 'director']

def combine_features(row):
    return row['genres'] + ' ' + row['keywords'] + ' ' + row['cast'] + ' ' + row['director']

for feature in features:
    df[feature] = df[feature].fillna('')

df_clip = df.apply(combine_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df_clip)

cosine_sim = cosine_similarity(count_matrix)

def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]

def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]

# Lets look for movies similar to Avatar
movie = 'Avatar'
movie_index = get_index_from_title(movie)
sim_movies = cosine_sim[movie_index]
sim_index_movies = []

for (count, index) in enumerate(sim_movies):
    count_index = (count, index)
    sim_index_movies.append(count_index)

sorted_sim_movies = sorted(sim_index_movies, key=lambda x:x[1], reverse=True)[1:]

print('Top 5 Similar Movies to Avatar: \n')
for i in range(5):
    similar_movie = get_title_from_index(sorted_sim_movies[i][0])
    print(similar_movie)

