from pydantic import BaseModel

class SKU(BaseModel):
    sku: str

class Keywords(BaseModel):
    search_keywords: str