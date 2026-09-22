from fastapi import FastAPI
from .routes import collections, documents, search

app = FastAPI(title="VectorForge API")
from vectorforge import db as vf_db

# ensure DB
vf_db.init_db()

# `collections`, `documents`, `search` are APIRouter objects exported from routes.__init__
app.include_router(collections, prefix="/api/v1")
app.include_router(documents, prefix="/api/v1")
app.include_router(search, prefix="/api/v1")
