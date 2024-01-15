import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

"""
# My first notebook

We want to display KMeans for different k.
"""

random = st.checkbox("Generate new data?")

@st.cache()
def create_data():
    X, y = make_blobs(
        n_samples=300,
        n_features=2,
        centers=6,
        cluster_std=1,
        random_state=42 if not random else np.random.randint(1, 1000),
    )

    return X, y

X, y = create_data()

"""
## Data

Visualization of data
"""

fig = plt.figure(figsize=(15, 8))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1")
plt.xlabel("x")
plt.ylabel("y")
plt.title("300 data points, 6 cluster")

st.pyplot(fig)

"""
## KMeans

Now we cluster the data with Kmans
"""

k = st.slider(
    "Please choose the number of clusters (k)", 
    min_value=1, max_value=10, step=1
)

run_kmeans = st.button("Start Clustering")

if run_kmeans:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    fig = plt.figure(figsize=(15, 8))

    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=100, alpha=0.6, cmap="Set1")
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="k",
        alpha=0.6,
        s=300,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"300 data points, {k} cluster")

    st.pyplot(fig)
