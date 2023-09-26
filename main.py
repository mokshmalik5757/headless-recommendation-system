from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

recommended_products = {
        "products": {
            "total_count": 5,
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
                },
                {
                    "id": 4226,
                    "attribute_set_id": 16,
                    "name": "SYMPHONY OMNI SUITE S 220MM",
                    "type_id": "simple",
                    "stock_status": "IN_STOCK",
                    "__typename": "SimpleProduct",
                    "description": {
                        "html": ""
                    },
                    "short_description": {
                        "html": "If the toilet and tank is fused together without any joints then the toilet is known as one piece toilet or Omni suites. It Comes in P Trap / S Trap ? 300, 220 & 110mm trap distance"
                    },
                    "meta_title": None,
                    "meta_description": None,
                    "sku": "SW_034",
                    "uid": "NDIyNg==",
                    "categories": [
                        {
                            "name": "Sanitaryware",
                            "url_key": "sanitaryware"
                        },
                        {
                            "name": "Omni Suites",
                            "url_key": "omni-suites"
                        }
                    ],
                    "image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jis0101b.jpg",
                        "label": "SYMPHONY OMNI SUITE S 220MM"
                    },
                    "small_image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jis0101b.jpg",
                        "label": "SYMPHONY OMNI SUITE S 220MM"
                    },
                    "media_gallery": [
                        {
                            "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jis0101b.jpg",
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
                                "value": 13990,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 13990,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        },
                        "maximum_price": {
                            "regular_price": {
                                "value": 13990,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 13990,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        }
                    }
                },
                {
                    "id": 4221,
                    "attribute_set_id": 16,
                    "name": "NEO Slim Single flushtank wt TOPKnob WH",
                    "type_id": "simple",
                    "stock_status": "IN_STOCK",
                    "__typename": "SimpleProduct",
                    "description": {
                        "html": "comes Smart sleek Durable body provides free \r\n \r\n  The shown are indicative may from actual"
                    },
                    "short_description": {
                        "html": ""
                    },
                    "meta_title": None,
                    "meta_description": None,
                    "sku": "SW_029",
                    "uid": "NDIyMQ==",
                    "categories": [
                        {
                            "name": "Sanitaryware",
                            "url_key": "sanitaryware"
                        },
                        {
                            "name": "Flush Tank",
                            "url_key": "flush-tank"
                        }
                    ],
                    "image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/p/2/p2811pw_flair.jpg",
                        "label": "NEO Slim Single flushtank wt TOPKnob WH"
                    },
                    "small_image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/p/2/p2811pw_flair.jpg",
                        "label": "NEO Slim Single flushtank wt TOPKnob WH"
                    },
                    "media_gallery": [
                        {
                            "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/p/2/p2811pw_flair.jpg",
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
                                "value": 1440,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 1440,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        },
                        "maximum_price": {
                            "regular_price": {
                                "value": 1440,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 1440,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        }
                    }
                },
                {
                    "id": 4204,
                    "attribute_set_id": 16,
                    "name": "SPIRIT SINK MIXER (WALL MOUNTED)",
                    "type_id": "simple",
                    "stock_status": "IN_STOCK",
                    "__typename": "SimpleProduct",
                    "description": {
                        "html": "handle both temperature pressure the This the common of of in today. Disclaimer finishes here only and differ the product."
                    },
                    "short_description": {
                        "html": ""
                    },
                    "meta_title": None,
                    "meta_description": None,
                    "sku": "SW_012",
                    "uid": "NDIwNA==",
                    "categories": [
                        {
                            "name": "Faucets & Fittings",
                            "url_key": "faucets-fittings"
                        },
                        {
                            "name": "Spirit",
                            "url_key": "spirit"
                        }
                    ],
                    "image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jit0204c.jpg",
                        "label": "SPIRIT SINK MIXER (WALL MOUNTED)"
                    },
                    "small_image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jit0204c.jpg",
                        "label": "SPIRIT SINK MIXER (WALL MOUNTED)"
                    },
                    "media_gallery": [
                        {
                            "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jit0204c.jpg",
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
                                "value": 5290,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 5290,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        },
                        "maximum_price": {
                            "regular_price": {
                                "value": 5290,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 5290,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        }
                    }
                },
                {
                    "id": 4197,
                    "attribute_set_id": 16,
                    "name": "ELITE 3000 PLAIN SPOUT",
                    "type_id": "simple",
                    "stock_status": "IN_STOCK",
                    "__typename": "SimpleProduct",
                    "description": {
                        "html": "handle both temperature pressure the This the common of of in today. Disclaimer finishes here only and differ the product."
                    },
                    "short_description": {
                        "html": ""
                    },
                    "meta_title": "Buy Online ELITE 3000 PLAIN SPOUT - Johnson Bathroom",
                    "meta_description": "Find best in class Faucets & Fittings from Johnson Bathroom`s. Buy online ELITE 3000 PLAIN SPOUT, @affordable price, Visit our nearest store or query online.",
                    "sku": "SW_005",
                    "uid": "NDE5Nw==",
                    "categories": [
                        {
                            "name": "Faucets & Fittings",
                            "url_key": "faucets-fittings"
                        },
                        {
                            "name": "Elite 3000",
                            "url_key": "elite-3000"
                        }
                    ],
                    "image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jit0012.jpg",
                        "label": "ELITE 3000 PLAIN SPOUT"
                    },
                    "small_image": {
                        "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jit0012.jpg",
                        "label": "ELITE 3000 PLAIN SPOUT"
                    },
                    "media_gallery": [
                        {
                            "url": "http://clouddeploy.in/media/catalog/product/cache/e4378c870fe0161e88bb87f253094d47/j/i/jit0012.jpg",
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
                                "value": 2260,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 2260,
                                "currency": "INR"
                            },
                            "discount": {
                                "amount_off": 0,
                                "percent_off": 0
                            }
                        },
                        "maximum_price": {
                            "regular_price": {
                                "value": 2260,
                                "currency": "INR"
                            },
                            "final_price": {
                                "value": 2260,
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