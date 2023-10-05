from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import api_router

app = FastAPI(title="Recommendation system API", debug = True)

app.include_router(api_router.router)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)