from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

recommended_products = {
        "products": {
            "total_count": 1,
            "items": [
                {
                    "id": 4231,
                    "attribute_set_id": 16,
                    "name": "PARIS BASIN WH ? P",
                    "type_id": "simple",
                    "stock_status": "OUT_OF_STOCK",
                    "__typename": "SimpleProduct",
                    "description": {
                        "html": ""
                    },
                    "short_description": {
                        "html": "Compact Design, Longer Durability,High Quality glazing for glossy and easy to clean surfaces"
                    },
                    "meta_title": None,
                    "meta_description": None,
                    "sku": "SW_039",
                    "uid": "NDIzMQ==",
                    "categories": [
                        {
                            "name": "Sanitaryware",
                            "url_key": "sanitaryware"
                        },
                        {
                            "name": "Wash Basin",
                            "url_key": "wash-basin"
                        }
                    ],
                    "image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jis0105b_1.jpg",
                        "label": "PARIS BASIN WH ? P"
                    },
                    "small_image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jis0105b_1.jpg",
                        "label": "PARIS BASIN WH ? P"
                    },
                    "media_gallery": [
                        {
                            "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jis0105b_1.jpg",
                            "label": None
                        }
                    ],
                    "related_products": [],
                    "upsell_products": [],
                    "crosssell_products": [],
                    "options": None,
                    "price_range": {
                        "minimum_price": {
                            "regular_price": {
                                "value": 1890,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 1890,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        },
                        "maximum_price": {
                            "regular_price": {
                                "value": 1890,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 1890,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        }
                    }
                }
            ]
        }
    }


app = FastAPI(title="Recommendation system API", debug = True)

# origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags= ["recommended_products"], status_code=status.HTTP_200_OK)
async def fetch_user_id():
    try:
        return recommended_products
    except Exception as e:
        return {"error": str(e)}