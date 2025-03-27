import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from pgvector.sqlalchemy import HALFVEC, BIT, Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import create_engine, Text, Index, insert

def create_synthetic_vectors(num_vectors, dimensions):
    return np.random.rand(num_vectors, dimensions).astype(np.float32)

def int8_quantization(vectors):
    min_val = vectors.min()
    max_val = vectors.max()
    scale = 127.0 / max(abs(min_val), abs(max_val))
    scaled = vectors * scale
    scaled = np.round(scaled)
    clipped = np.clip(scaled, -128, 127)
    int8_embeddings = clipped.astype(np.int8)
    return int8_embeddings

def binary_quantization_simple_threshold(vectors):
    return (vectors > 0).astype(np.bool)

def binary_quantization_vector_mean(vectors):
    mean = vectors.mean(axis=1)
    return (vectors > mean[:, np.newaxis]).astype(np.bool)

def binary_quantization_vector_median(vectors):
    median = np.median(vectors, axis=1)
    return (vectors > median[:, np.newaxis]).astype(np.bool)


example_vectors = create_synthetic_vectors(10, 1024)

print(example_vectors)

int8_quantized = int8_quantization(example_vectors)

print(int8_quantized)

binary_quantized = binary_quantization_simple_threshold(example_vectors)

print(binary_quantized)

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

example_corpus = ["Vector quantization is the bread and butter of machine learning engineers"]
embeddings = model.encode(example_corpus)

def quantize_embeddings(embeddings, precision="int8"):
    #Available precisions: int8, uint8, binary, ubinary
    try:
        return quantize_embeddings(
            embeddings,
            precision=f"{precision}",
        )
    except Exception as e:
        print(f"Error quantizing embeddings: {e}")
        return None



class Base(DeclarativeBase):
    pass

class Fp32Embedding(Base):
    __tablename__ = "fp32_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Vector] = mapped_column(Vector(1024))
    
engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")

Base.metadata.create_all(engine)

index = Index(
    'idx_fp32_vector',
    Fp32Embedding.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'vector': 'vector_l2_ops'}
)
index.create(engine)

session = Session(engine)
session.add(Fp32Embedding(content="Vector quantization is the bread and butter of machine learning engineers", embedding=embeddings))
session.commit()


class Fp16Embedding(Base):
    __tablename__ = "fp16_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Vector] = mapped_column(HALFVEC(1024))

index = Index(
    'idx_fp16_vector',
    Fp16Embedding.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'vector': 'halfvec_l2_ops'}
)
index.create(engine)
engine.execute(Fp16Embedding(content="Vector quantization is the bread and butter of machine learning engineers", embedding=embeddings))
engine.commit()

class BinaryEmbedding(Base):
    __tablename__ = "binary_embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Vector] = mapped_column(BIT(1024))

index = Index(
    'idx_binary_vector',
    BinaryEmbedding.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'vector': 'bitvec_hamming_ops'}
)











