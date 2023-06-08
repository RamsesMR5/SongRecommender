import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spotify")
st.title("Song Recommender")


index = st.slider('Pick a number', 0,99)

df1 = pd.read_csv("data/ram-test.csv")
df2 = pd.read_csv("data/ram-test2.csv")
df3 = pd.read_csv("data/ram-test3.csv")

df = pd.concat([df1, df2], ignore_index=True)

df.drop('Unnamed: 0', inplace=True, axis=1)
df3.drop('Unnamed: 0', inplace=True, axis=1)

features = df.drop(columns=['song_id'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)

features = df.drop(columns=['song_id'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)

k = 3  
X= features
model = KMeans(n_clusters=3)  
model.fit(X)
predictions = model.predict(X)

input_data = df3
dataset = df

def get_recommendation(index, kmeans_model, dataset, default_value=None):
    if index < 0 or index >= len(dataset):

        return default_value
    
    input_data = dataset.iloc[[index]]
    input_cluster = kmeans_model.predict(input_data.drop('song_id', axis=1))
    
    cluster_indices = np.where(kmeans_model.labels_ == input_cluster)[0]
    
    if len(cluster_indices) == 0:

        return default_value
    
    song_id = dataset.iloc[cluster_indices]['song_id'].values[0]
    
    return song_id

k = 3  # Number of clusters
kmeans_model = KMeans(n_clusters=k)
kmeans_model.fit(dataset.drop('song_id', axis=1))


recommendation = get_recommendation(index, kmeans_model, dataset)

if st.button(f"go to song"):    
    os.system(f"open https://open.spotify.com/track/{recommendation}")




