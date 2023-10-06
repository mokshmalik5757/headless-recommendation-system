from fastapi import APIRouter, status
import pandas as pd
import numpy as np
from app import schemas
from .functions import recommend_products_by_query, recommend_related_products

router = APIRouter()

@router.post("/related_products/", tags= ["related_products"], status_code=status.HTTP_200_OK)
async def get_related_products(sku: schemas.SKU, num_recommendations: int = 10):
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

    
@router.post("/recommended_products/", tags= ["recommended_products"], status_code=status.HTTP_200_OK)
async def get_recommended_products(search_keywords: schemas.Keywords, num_recommendations: int = 10):
    try:
        
        query = search_keywords.search_keywords
        
        recommendations = recommend_products_by_query(query, num_recommendations)
        recommendations = recommendations.replace(['Unknown', np.nan], None)
        
        return {"message": "Success", "result": recommendations.drop(['text_features','text_embeddings'], axis=1).to_dict(orient='records'), "status": str(status.HTTP_200_OK)}

    except Exception:
        return {"message": f"No product found for keywords {search_keywords.search_keywords}", "result": [], "status": str(status.HTTP_400_BAD_REQUEST)} 