import { sql, eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import { int8Embedding } from "./db/schema";

function createSyntheticVectors(
  num_vectors: number = 3,
  dimensions: number = 1024
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

function int8Quantization(vectors: Float32Array[]): Int8Array[] {
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

async function main() {
  const pool = new Pool({
    connectionString: "postgresql://postgres:postgres@localhost:5432/pgvector",
  });
  const db = drizzle(pool);

  const vectors = createSyntheticVectors(1, 1024);
  const quantizedVectors = int8Quantization(vectors);

  await db.insert(int8Embedding).values({
    embedding: sql`ARRAY[${sql.raw(Array.from(quantizedVectors[0]).join(","))}]::vector`,
  });

  const result = await db
    .select()
    .from(int8Embedding)
    .where(eq(int8Embedding.id, 1))
    .limit(1);

  console.log(result);

  await pool.end();
}

main().catch(console.error);
