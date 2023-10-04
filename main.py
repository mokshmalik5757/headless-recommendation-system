from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
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
from pydantic import BaseModel

app = FastAPI(title="Recommendation system API", debug = True)

class SKU(BaseModel):
    sku: str

class Keywords(BaseModel):
    search_keywords: list[str]

product_embeddings = np.load(r"./product_embeddings.npy")
products_df = pd.read_csv(r"./headless data preprocessed")

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model = AutoModel.from_pretrained("thenlper/gte-small")

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

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/related_products/", tags= ["related_products"], status_code=status.HTTP_200_OK)
async def get_related_products(sku: SKU, num_recommendations: int = 10):
    try:
        sku = sku.sku # type: ignore
        num_recommendations = num_recommendations
        
        related_products_cluster_5 = recommend_related_products(5, sku, num_recommendations)
        related_products_cluster_6 = recommend_related_products(6, sku, num_recommendations)

        result_df = pd.concat([related_products_cluster_5, related_products_cluster_6], axis=0, ignore_index=True).replace(['Unknown', np.nan], None)
        related_products = result_df.drop(['text_features','text_embeddings','cluster_label'], axis=1).to_dict(orient='records')
        return {"message": "Success", "result": related_products, "status": str(status.HTTP_200_OK)}
    except Exception:
        return {"message": f"No product found for the SKU {sku}", "result": [], "status": str(status.HTTP_400_BAD_REQUEST)}

    
@app.post("/recommended_products/", tags= ["recommended_products"], status_code=status.HTTP_200_OK)
async def get_recommended_products(search_keywords: Keywords, num_recommendations: int = 10):
    try:
        final_df = pd.DataFrame()
        query = search_keywords.search_keywords
        for item in query:
            recommendations = recommend_products_by_query(item, num_recommendations)
            final_df = pd.concat([final_df, recommendations], axis=0)
            final_df = final_df.sample(frac=1).replace(['Unknown', np.nan], None)
        
        return {"message": "Success", "result": final_df.drop(['text_features','text_embeddings'], axis=1).to_dict(orient='records'), "status": str(status.HTTP_200_OK)}

    except Exception:
        return {"message": f"No product found for keywords {search_keywords.search_keywords}","result": [], "status": str(status.HTTP_400_BAD_REQUEST)} 