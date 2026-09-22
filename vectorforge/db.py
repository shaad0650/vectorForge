from __future__ import annotations
import datetime
from pathlib import Path

BASE = Path("data")
BASE.mkdir(parents=True, exist_ok=True)

try:
    from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
    from sqlalchemy.orm import declarative_base, sessionmaker

    DB_PATH = BASE / "vectorforge.db"
    ENGINE = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=ENGINE)
    Base = declarative_base()


    class Collection(Base):
        __tablename__ = "collections"
        id = Column(String, primary_key=True, index=True)
        name = Column(String, nullable=False)
        description = Column(Text, nullable=True)
        embedding_model = Column(String, nullable=True)
        embedding_dimension = Column(Integer, nullable=True)
        distance_metric = Column(String, nullable=True)
        vector_count = Column(Integer, nullable=True)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.datetime.utcnow)


    class IndexMeta(Base):
        __tablename__ = "indexes"
        id = Column(String, primary_key=True, index=True)
        collection_id = Column(String, nullable=False, index=True)
        index_type = Column(String, nullable=False)
        index_path = Column(String, nullable=False)
        parameters_json = Column(Text, nullable=True)
        vector_count = Column(Integer, nullable=True)
        dimension = Column(Integer, nullable=True)
        build_time_ms = Column(Integer, nullable=True)
        memory_bytes = Column(Integer, nullable=True)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        status = Column(String, default="built")


    def init_db():
        Base.metadata.create_all(bind=ENGINE)


    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    DB_AVAILABLE = True
except Exception:
    # SQLAlchemy not installed; provide stubs so tests can run without DB
    SessionLocal = None
    Collection = None
    IndexMeta = None

    def init_db():
        return

    def get_db():
        yield None

    DB_AVAILABLE = False
