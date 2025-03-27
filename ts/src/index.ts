import { sql } from "drizzle-orm";
import { pgTable, serial, text, vector, index } from "drizzle-orm/pg-core";
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

const fp32Embedding = pgTable(
  "fp32_embeddings",
  {
    id: serial("id").primaryKey(),
    content: text("content").notNull(),
    embedding: vector("embedding", { dimensions: 1024 }).notNull(),
  },
  (table) => [
    index("l2_index").using("hnsw", table.embedding.op("vector_l2_ops")),
    // index('ip_index').using('hnsw', table.embedding.op('vector_ip_ops'))
    // index('cosine_index').using('hnsw', table.embedding.op('vector_cosine_ops'))
  ]
);

const int8Embedding = pgTable(
  "int8_embeddings",
  {
    id: serial("id").primaryKey(),
    content: text("content").notNull(),
    embedding: vector("embedding", { dimensions: 1024 }).notNull(),
  },
  (table) => [
    index("l2_index").using("hnsw", table.embedding.op("vector_l2_ops")),
    // index('ip_index').using('hnsw', table.embedding.op('vector_ip_ops'))
    // index('cosine_index').using('hnsw', table.embedding.op('vector_cosine_ops'))
  ]
);

const binaryEmbedding = pgTable(
  "binary_embeddings",
  {
    id: serial("id").primaryKey(),
    content: text("content").notNull(),
    embedding: vector("embedding", { dimensions: 1024 }).notNull(),
  },
  (table) => [
    index("hamming_index").using("hnsw", table.embedding.op("bit_hamming_ops")),
    //index("l1_index").using("hnsw", table.embedding.op("vector_l1_ops")),
    // index('bit_jaccard_index').using('hnsw', table.embedding.op('bit_jaccard_ops'))
  ]
);

export function createSyntheticVectors(
  num_vectors: number,
  dimensions: number
): Float32Array[] {
  const vectors: Float32Array[] = [];
  for (let i = 0; i < num_vectors; i++) {
    const vector = new Float32Array(dimensions);
    for (let j = 0; j < dimensions; j++) {
      vector[j] = Math.random();
    }
    vectors.push(vector);
  }
  return vectors;
}

export async function insertVectors(vectors: Float32Array[]) {
  const pool = new Pool({
    connectionString: "postgresql://postgres:postgres@localhost:5432/postgres",
  });
  const db = drizzle(pool);

  const quantizedInt8 = int8Quantization(vectors);
  await db.insert(int8Embedding).values(
    quantizedInt8.map((vector, i) => ({
      content: `Example ${i}`,
      embedding: sql`[${Array.from(vector).join(",")}]::vector`,
    }))
  );
  await pool.end();
}

export function int8Quantization(vectors: Float32Array[]): Int8Array[] {
  const [minVal, maxVal] = vectors.reduce(
    ([min, max], vector) => [
      Math.min(min, ...vector),
      Math.max(max, ...vector),
    ],
    [Infinity, -Infinity]
  );
  const scale = 127.0 / Math.max(Math.abs(minVal), Math.abs(maxVal));
  return vectors.map((vector) => {
    const result = new Int8Array(vector.length);
    for (let i = 0; i < vector.length; i++) {
      result[i] = Math.max(-127, Math.min(127, Math.round(vector[i] * scale)));
    }
    return result;
  });
}

export function simpleBinaryQuantization(vectors: Float32Array[]): number[][] {
  return vectors.map((vector) =>
    Array.from(vector).map((val) => (val > 0 ? 1 : 0))
  );
}
