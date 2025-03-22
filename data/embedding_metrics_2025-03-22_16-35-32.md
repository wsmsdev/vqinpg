# Embedding Performance Metrics

Generated: 2025-03-22 16:35:32

## Performance Comparison

| Metric                  | FP32               | INT8               | Binary              |
| ----------------------- | ------------------ | ------------------ | ------------------- |
| Total Vectors           | 3633               | 3633               | 3633                |
| Dimensions              | 384                | 384                | 384                 |
| Insert Time (s)         | 10.414243698120117 | 8.601150035858154  | 8.108979940414429   |
| Index Creation Time (s) | 0.5502369403839111 | 0.5390281677246094 | 0.12845683097839355 |
| Table Size (MB)         | 10.12              | 10.12              | 6.37                |
| Index Size (MB)         | 7.03               | 7.03               | 0.05                |

## FP32 Embeddings

- **Vectors:** 3633
- **Dimensions:** 384
- **Insert Time:** 10.41 seconds
- **Index Creation Time:** 0.55 seconds
- **Table Size:** 10.12 MB
- **Index Size:** 7.03 MB

## INT8 Embeddings

- **Vectors:** 3633
- **Dimensions:** 384
- **Insert Time:** 8.60 seconds
- **Index Creation Time:** 0.54 seconds
- **Table Size:** 10.12 MB
- **Index Size:** 7.03 MB

## BINARY Embeddings

- **Vectors:** 3633
- **Dimensions:** 384
- **Insert Time:** 8.11 seconds
- **Index Creation Time:** 0.13 seconds
- **Table Size:** 6.37 MB
- **Index Size:** 0.05 MB
