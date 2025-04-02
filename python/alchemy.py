from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import Session
import numpy as np

def create_synthetic_vectors(num_vectors=3, dimensions=1024):
    vectors = []
    for i in range(num_vectors):
        vector = np.random.rand(dimensions).astype(np.float32)
        vectors.append(vector)
    return vectors

class Base(DeclarativeBase):
    pass

class Fp16Embedding(Base):
    __tablename__ = "fp16_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    embedding: Mapped[Vector] = mapped_column(HALFVEC(1024))
    
    def __repr__(self):
        return f"Fp16Embedding(id={self.id})"

engine = create_engine("postgresql://postgres:postgres@localhost:5432/pgvector")

Base.metadata.create_all(engine)

index = Index(
    'idx_fp16_vector',
    Fp16Embedding.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'halfvec_l2_ops'}
)

vectors = create_synthetic_vectors()

with Session(engine) as session:
    session.add(Fp16Embedding(id=1, embedding=vectors[0]))
    session.commit()
    
quantized_vector = session.query(Fp16Embedding).filter(Fp16Embedding.id == 1).first().embedding

print(quantized_vector)
