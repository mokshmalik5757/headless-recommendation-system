import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

product_embeddings = np.load("./app/static/product_embeddings.npy")
products_df = pd.read_csv("./app/static/headless data preprocessed.csv")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

encoder = OneHotEncoder(handle_unknown="ignore")
encoded_features = encoder.fit_transform(
    products_df[["appliances_color"]]).toarray() # type: ignore

def recommend_related_products(num_clusters, sku, num_recommendations=5):
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init='auto', random_state=42)
    products_df["cluster_label"] = kmeans.fit_predict(product_embeddings)
    cluster_label = products_df.loc[
        products_df["sku"] == sku, "cluster_label"
    ].iloc[0] # type: ignore
    related_products = products_df[products_df["cluster_label"] == cluster_label]

    if len(related_products) > num_recommendations:
        related_products = related_products.sample(n=num_recommendations)

    return related_products

def embed_text(text):
    tokens = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt", max_length=128
    )
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def correct_spelling(query):
    return str(TextBlob(query).correct())

def recommend_products_by_query(query, num_recommendations=5):
    corrected_query = correct_spelling(query)

    query_embedding = embed_text(corrected_query)

    query_cat_features = encoder.transform(
        [["Unknown"]] # type: ignore
    ).toarray() # type: ignore

    query_vector = np.concatenate([query_embedding, query_cat_features], axis=1)

    cosine_sim_query = cosine_similarity(query_vector, product_embeddings)

    sim_scores = list(enumerate(cosine_sim_query[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:num_recommendations]
    product_indices = [i[0] for i in sim_scores]

    return products_df.iloc[product_indices]
