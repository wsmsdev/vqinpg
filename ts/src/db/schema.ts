import { pgTable, serial, text, vector, index } from "drizzle-orm/pg-core";

const int8Embedding = pgTable(
  "int8_embeddings",
  {
    id: serial("id").primaryKey(),
    embedding: vector("embedding", { dimensions: 1024 }).notNull(),
  },
  (table) => [
    {
      l2: index("l2_index").using("hnsw", table.embedding.op("vector_l2_ops")),
      //ip: index('ip_index').using('hnsw', table.embedding.op('vector_ip_ops'))
      //cosine: index('cosine_index').using('hnsw', table.embedding.op('vector_cosine_ops'))
    },
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
    {
      hamming: index("hamming_index").using(
        "hnsw",
        table.embedding.op("bit_hamming_ops")
      ),
      jaccard: index("jaccard_index").using(
        "hnsw",
        table.embedding.op("bit_jaccard_ops")
      ),
    },
  ]
);

export { int8Embedding, binaryEmbedding };
