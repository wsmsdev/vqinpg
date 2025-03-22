import pandas as pd
import psycopg2
import os
import time
import numpy as np
from dotenv import load_dotenv
import logging
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from tqdm import tqdm


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def connect_to_db():
    load_dotenv()
    conn = psycopg2.connect(
        host="localhost",
        database=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        port=5432,
    )
    return conn


def load_evaluation_data(logger):
    logger.info("Loading MTEB evaluation data")
    test_dataset = load_dataset("mteb/nfcorpus", name="queries")
    test_queries = test_dataset["queries"]

    query_ids = [q["_id"] for q in test_queries]
    query_texts = [q["text"] for q in test_queries]

    logger.info(f"Loaded {len(query_ids)} test queries")
    return query_ids, query_texts


def encode_queries(query_texts, dimensions, logger):
    logger.info("Encoding test queries")
    model = SentenceTransformer(
        "mixedbread-ai/mxbai-embed-xsmall-v1", truncate_dim=dimensions
    )

    query_embeddings = model.encode(query_texts, convert_to_numpy=True)
    logger.info(f"Encoded {len(query_embeddings)} queries")

    query_embeddings_binary = (query_embeddings > 0).astype(bool)

    # For halfvec (fp16), we'll just use the fp32 embeddings directly
    # pgvector will cast them to fp16 when using the halfvec column
    logger.info("Generated fp32, binary embeddings for queries")
    return query_embeddings, query_embeddings_binary


def run_search_queries(
    conn,
    query_ids,
    query_embeddings,
    query_embeddings_binary,
    logger,
    k=10,
    num_queries=100,
):
    logger.info(
        f"Running {num_queries} search queries (retrieving top {k} results each)"
    )

    if num_queries < len(query_ids):
        indices = np.random.choice(len(query_ids), num_queries, replace=False)
        subset_query_ids = [query_ids[i] for i in indices]
        subset_query_embeddings = query_embeddings[indices]
        subset_query_embeddings_binary = query_embeddings_binary[indices]
    else:
        subset_query_ids = query_ids
        subset_query_embeddings = query_embeddings
        subset_query_embeddings_binary = query_embeddings_binary

    results = {
        "fp32": {"timings": [], "results": []},
        "halfvec": {"timings": [], "results": []},
        "bit": {"timings": [], "results": []},
    }

    cursor = conn.cursor()

    logger.info("Testing FP32 search")
    for i, (query_id, query_embedding) in enumerate(
        tqdm(zip(subset_query_ids, subset_query_embeddings))
    ):
        vector_str = "[" + ",".join([str(float(val)) for val in query_embedding]) + "]"

        start_time = time.time()
        cursor.execute(
            f"""
            SELECT id, text, embedding <-> %s::vector AS distance
            FROM embeddings_fp32
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """,
            (vector_str, vector_str, k),
        )
        end_time = time.time()

        query_results = cursor.fetchall()
        results["fp32"]["timings"].append(end_time - start_time)
        results["fp32"]["results"].append(
            {
                "query_id": query_id,
                "retrieved": [
                    {"id": row[0], "text": row[1], "distance": row[2]}
                    for row in query_results
                ],
            }
        )

    logger.info("Testing HalfVec search")
    for i, (query_id, query_embedding) in enumerate(
        tqdm(zip(subset_query_ids, subset_query_embeddings))
    ):
        vector_str = "[" + ",".join([str(float(val)) for val in query_embedding]) + "]"

        start_time = time.time()
        cursor.execute(
            f"""
            SELECT id, text, embedding <-> %s::halfvec AS distance
            FROM embeddings_halfvec
            ORDER BY embedding <-> %s::halfvec
            LIMIT %s
        """,
            (vector_str, vector_str, k),
        )
        end_time = time.time()

        query_results = cursor.fetchall()
        results["halfvec"]["timings"].append(end_time - start_time)
        results["halfvec"]["results"].append(
            {
                "query_id": query_id,
                "retrieved": [
                    {"id": row[0], "text": row[1], "distance": row[2]}
                    for row in query_results
                ],
            }
        )

    logger.info("Testing Bit search")
    for i, (query_id, query_embedding) in enumerate(
        tqdm(zip(subset_query_ids, subset_query_embeddings_binary))
    ):
        bit_string = "".join(["1" if val else "0" for val in query_embedding])
        start_time = time.time()
        cursor.execute(
            f"""
            SELECT id, text, embedding <~> %s::bit(384) AS distance
            FROM embeddings_bit
            ORDER BY embedding <~> %s::bit(384)
            LIMIT %s
        """,
            (bit_string, bit_string, k),
        )
        end_time = time.time()
        query_results = cursor.fetchall()
        results["bit"]["timings"].append(end_time - start_time)
        results["bit"]["results"].append(
            {
                "query_id": query_id,
                "retrieved": [
                    {"id": row[0], "text": row[1], "distance": row[2]}
                    for row in query_results
                ],
            }
        )

    cursor.close()
    return results


def calculate_performance_metrics(search_results, logger):
    metrics = {}

    for emb_type, data in search_results.items():
        timings = data["timings"]
        metrics[emb_type] = {
            "avg_query_time": np.mean(timings),
            "p50_query_time": np.percentile(timings, 50),
            "p90_query_time": np.percentile(timings, 90),
            "p95_query_time": np.percentile(timings, 95),
            "p99_query_time": np.percentile(timings, 99),
            "min_query_time": np.min(timings),
            "max_query_time": np.max(timings),
            "queries_per_second": 1.0 / np.mean(timings),
            "total_queries": len(timings),
        }

    logger.info("Calculated performance metrics")
    return metrics


def calculate_result_agreement(search_results, logger):
    agreement_metrics = {"fp32_vs_halfvec": [], "fp32_vs_bit": [], "halfvec_vs_bit": []}

    num_queries = len(search_results["fp32"]["results"])

    for i in range(num_queries):
        fp32_ids = [
            item["id"] for item in search_results["fp32"]["results"][i]["retrieved"]
        ]
        halfvec_ids = [
            item["id"] for item in search_results["halfvec"]["results"][i]["retrieved"]
        ]
        bit_ids = [
            item["id"] for item in search_results["bit"]["results"][i]["retrieved"]
        ]
        fp32_halfvec_jaccard = len(set(fp32_ids).intersection(set(halfvec_ids))) / len(
            set(fp32_ids).union(set(halfvec_ids))
        )
        fp32_bit_jaccard = len(set(fp32_ids).intersection(set(bit_ids))) / len(
            set(fp32_ids).union(set(bit_ids))
        )
        halfvec_bit_jaccard = len(set(halfvec_ids).intersection(set(bit_ids))) / len(
            set(halfvec_ids).union(set(bit_ids))
        )
        agreement_metrics["fp32_vs_halfvec"].append(fp32_halfvec_jaccard)
        agreement_metrics["fp32_vs_bit"].append(fp32_bit_jaccard)
        agreement_metrics["halfvec_vs_bit"].append(halfvec_bit_jaccard)

    for key in agreement_metrics:
        agreement_metrics[key] = np.mean(agreement_metrics[key])

    logger.info("Calculated result agreement metrics")
    return agreement_metrics


def evaluate_ranking_correlation(search_results, logger):
    correlation_metrics = {
        "fp32_vs_halfvec": [],
        "fp32_vs_bit": [],
        "halfvec_vs_bit": [],
    }

    num_queries = len(search_results["fp32"]["results"])

    for i in range(num_queries):
        fp32_ids = [
            item["id"] for item in search_results["fp32"]["results"][i]["retrieved"]
        ]
        halfvec_ids = [
            item["id"] for item in search_results["halfvec"]["results"][i]["retrieved"]
        ]
        bit_ids = [
            item["id"] for item in search_results["bit"]["results"][i]["retrieved"]
        ]
        fp32_ranks = {doc_id: rank for rank, doc_id in enumerate(fp32_ids)}
        halfvec_ranks = {doc_id: rank for rank, doc_id in enumerate(halfvec_ids)}
        bit_ranks = {doc_id: rank for rank, doc_id in enumerate(bit_ids)}
        common_fp32_halfvec = set(fp32_ids).intersection(set(halfvec_ids))
        common_fp32_bit = set(fp32_ids).intersection(set(bit_ids))
        common_halfvec_bit = set(halfvec_ids).intersection(set(bit_ids))

        if len(common_fp32_halfvec) > 1:
            fp32_halfvec_ranks = (
                [fp32_ranks[doc] for doc in common_fp32_halfvec],
                [halfvec_ranks[doc] for doc in common_fp32_halfvec],
            )
            corr, _ = spearmanr(*fp32_halfvec_ranks)
            correlation_metrics["fp32_vs_halfvec"].append(corr)

        if len(common_fp32_bit) > 1:
            fp32_bit_ranks = (
                [fp32_ranks[doc] for doc in common_fp32_bit],
                [bit_ranks[doc] for doc in common_fp32_bit],
            )
            corr, _ = spearmanr(*fp32_bit_ranks)
            correlation_metrics["fp32_vs_bit"].append(corr)

        if len(common_halfvec_bit) > 1:
            halfvec_bit_ranks = (
                [halfvec_ranks[doc] for doc in common_halfvec_bit],
                [bit_ranks[doc] for doc in common_halfvec_bit],
            )
            corr, _ = spearmanr(*halfvec_bit_ranks)
            correlation_metrics["halfvec_vs_bit"].append(corr)

    for key in correlation_metrics:
        values = [v for v in correlation_metrics[key] if not np.isnan(v)]
        correlation_metrics[key] = np.mean(values) if values else float("nan")

    logger.info("Calculated ranking correlation metrics")
    return correlation_metrics


def save_metrics_to_markdown(
    performance_metrics, agreement_metrics, correlation_metrics, logger
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./data/search_metrics_{timestamp}.md"

    with open(filename, "w") as f:
        f.write("# Vector Search Performance Metrics\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Performance Metrics\n\n")
        f.write("| Metric | FP32 | HalfVec (fp16) | Bit |\n")
        f.write("|--------|------|----------------|-----|\n")

        perf_metrics_to_display = [
            ("Avg Query Time (ms)", lambda x: x["avg_query_time"] * 1000),
            ("p50 Query Time (ms)", lambda x: x["p50_query_time"] * 1000),
            ("p90 Query Time (ms)", lambda x: x["p90_query_time"] * 1000),
            ("p95 Query Time (ms)", lambda x: x["p95_query_time"] * 1000),
            ("p99 Query Time (ms)", lambda x: x["p99_query_time"] * 1000),
            ("Min Query Time (ms)", lambda x: x["min_query_time"] * 1000),
            ("Max Query Time (ms)", lambda x: x["max_query_time"] * 1000),
            ("Queries Per Second", lambda x: x["queries_per_second"]),
            ("Total Queries", lambda x: x["total_queries"]),
        ]

        for label, accessor in perf_metrics_to_display:
            fp32_val = accessor(performance_metrics["fp32"])
            halfvec_val = accessor(performance_metrics["halfvec"])
            bit_val = accessor(performance_metrics["bit"])

            if label.endswith("Per Second") or label.endswith("Queries"):
                f.write(
                    f"| {label} | {fp32_val:.2f} | {halfvec_val:.2f} | {bit_val:.2f} |\n"
                )
            else:
                f.write(
                    f"| {label} | {fp32_val:.3f} | {halfvec_val:.3f} | {bit_val:.3f} |\n"
                )

        f.write("\n## Result Agreement (Jaccard Similarity)\n\n")
        f.write("| Comparison | Jaccard Similarity |\n")
        f.write("|------------|--------------------|\n")
        f.write(
            f"| FP32 vs HalfVec (fp16) | {agreement_metrics['fp32_vs_halfvec']:.4f} |\n"
        )
        f.write(f"| FP32 vs Bit | {agreement_metrics['fp32_vs_bit']:.4f} |\n")
        f.write(
            f"| HalfVec (fp16) vs Bit | {agreement_metrics['halfvec_vs_bit']:.4f} |\n"
        )

        f.write("\n## Ranking Correlation (Spearman's Rho)\n\n")
        f.write("| Comparison | Correlation |\n")
        f.write("|------------|-------------|\n")
        f.write(
            f"| FP32 vs HalfVec (fp16) | {correlation_metrics['fp32_vs_halfvec']:.4f} |\n"
        )
        f.write(f"| FP32 vs Bit | {correlation_metrics['fp32_vs_bit']:.4f} |\n")
        f.write(
            f"| HalfVec (fp16) vs Bit | {correlation_metrics['halfvec_vs_bit']:.4f} |\n"
        )

        f.write("\n## Vector Type Comparison\n\n")
        f.write("| Vector Type | Precision | Bytes Per Dimension | Relative Size |\n")
        f.write("|-------------|-----------|---------------------|---------------|\n")
        f.write("| FP32 | 32-bit floating point | 4 bytes | 100% |\n")
        f.write("| HalfVec | 16-bit floating point | 2 bytes | 50% |\n")
        f.write("| Bit | 1-bit binary | 1/8 byte | 3.125% |\n")

        f.write("\n## Interpretation\n\n")

        fp32_qps = performance_metrics["fp32"]["queries_per_second"]
        halfvec_qps = performance_metrics["halfvec"]["queries_per_second"]
        bit_qps = performance_metrics["bit"]["queries_per_second"]

        halfvec_speedup = (halfvec_qps / fp32_qps - 1) * 100
        bit_speedup = (bit_qps / fp32_qps - 1) * 100

        f.write("### Performance\n\n")
        f.write(
            f"- HalfVec (fp16) provides a **{halfvec_speedup:.1f}%** speedup compared to FP32 vectors\n"
        )
        f.write(
            f"- Bit vectors provide a **{bit_speedup:.1f}%** speedup compared to FP32 vectors\n\n"
        )

        f.write("### Result Quality\n\n")

        fp32_halfvec_agreement = agreement_metrics["fp32_vs_halfvec"] * 100
        fp32_bit_agreement = agreement_metrics["fp32_vs_bit"] * 100

        f.write(
            f"- HalfVec (fp16) vectors return **{fp32_halfvec_agreement:.1f}%** of the same results as FP32 vectors\n"
        )
        f.write(
            f"- Bit vectors return **{fp32_bit_agreement:.1f}%** of the same results as FP32 vectors\n\n"
        )

        f.write("### Recommendations\n\n")

        if halfvec_speedup > 20 and fp32_halfvec_agreement > 90:
            f.write(
                "- **HalfVec (fp16) is recommended** for this workload: significant performance improvement with minimal impact on result quality\n"
            )
        elif halfvec_speedup > 0:
            f.write(
                "- HalfVec (fp16) provides some performance benefits with reasonable result quality\n"
            )

        if bit_speedup > 50 and fp32_bit_agreement > 70:
            f.write(
                "- **Bit vectors are recommended** for high-throughput scenarios where approximate results are acceptable\n"
            )
        elif bit_speedup > 0 and fp32_bit_agreement < 50:
            f.write(
                "- Bit vectors provide speed but with significant changes in results - consider using with reranking\n"
            )

    logger.info(f"Metrics saved to {filename}")
    return filename


def main():
    logger = setup_logging()

    try:
        query_ids, query_texts = load_evaluation_data(logger)
        dimensions = 384
        query_embeddings, query_embeddings_binary = encode_queries(
            query_texts, dimensions, logger
        )
        conn = connect_to_db()
        search_results = run_search_queries(
            conn,
            query_ids,
            query_embeddings,
            query_embeddings_binary,
            logger,
            k=10,
            num_queries=50,
        )

        performance_metrics = calculate_performance_metrics(search_results, logger)
        agreement_metrics = calculate_result_agreement(search_results, logger)
        correlation_metrics = evaluate_ranking_correlation(search_results, logger)

        metrics_file = save_metrics_to_markdown(
            performance_metrics, agreement_metrics, correlation_metrics, logger
        )

        logger.info("\n=== PERFORMANCE SUMMARY ===")
        for emb_type in performance_metrics:
            type_label = "HalfVec (fp16)" if emb_type == "halfvec" else emb_type.upper()
            logger.info(f"\n{type_label}:")
            logger.info(
                f"  Avg Query Time: {performance_metrics[emb_type]['avg_query_time']*1000:.3f} ms"
            )
            logger.info(
                f"  p99 Query Time: {performance_metrics[emb_type]['p99_query_time']*1000:.3f} ms"
            )
            logger.info(
                f"  Queries Per Second: {performance_metrics[emb_type]['queries_per_second']:.2f}"
            )

        logger.info("\n=== RESULT QUALITY SUMMARY ===")
        logger.info(
            f"  FP32 vs HalfVec (fp16) agreement: {agreement_metrics['fp32_vs_halfvec']:.4f}"
        )
        logger.info(f"  FP32 vs Bit agreement: {agreement_metrics['fp32_vs_bit']:.4f}")
        logger.info(f"\nDetailed metrics saved to: {metrics_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        if "conn" in locals():
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()
