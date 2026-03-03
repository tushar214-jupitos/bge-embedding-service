import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

model: Optional[SentenceTransformer] = None

QUERY_PREFIX = "Represent this sentence: "


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_name = os.getenv("MODEL_NAME", "BAAI/bge-large-en-v1.5")
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    yield
    model = None


app = FastAPI(
    title="BGE Embedding Service",
    version="1.0.0",
    lifespan=lifespan,
)


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    is_query: bool = Field(
        default=False,
        description="If True, prepends BGE query instruction prefix. Use True for search queries, False for documents being indexed.",
    )


class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    is_query: bool = Field(default=False)


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimensions: int


class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "dimensions": model.get_sentence_embedding_dimension() if model else None,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed_single(req: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if req.is_query:
        text = QUERY_PREFIX + text

    vector = model.encode(text, normalize_embeddings=True)
    return EmbedResponse(
        embedding=vector.tolist(),
        dimensions=len(vector),
    )


@app.post("/embed-batch", response_model=EmbedBatchResponse)
async def embed_batch(req: EmbedBatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    texts = [t.strip() for t in req.texts]
    if req.is_query:
        texts = [QUERY_PREFIX + t for t in texts]

    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return EmbedBatchResponse(
        embeddings=[v.tolist() for v in vectors],
        dimensions=vectors.shape[1],
        count=len(vectors),
    )
