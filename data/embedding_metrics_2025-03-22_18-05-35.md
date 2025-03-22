# Embedding Performance Metrics

Generated: 2025-03-22 18:05:35

## Performance Comparison

| Metric                  | FP32              | HALFVEC             | BIT                |
| ----------------------- | ----------------- | ------------------- | ------------------ |
| Total Vectors           | 3633              | 3633                | 3633               |
| Dimensions              | 384               | 384                 | 384                |
| Insert Time (s)         | 7.515048980712891 | 7.999542951583862   | 5.212824821472168  |
| Index Creation Time (s) | 0.539557933807373 | 0.41348910331726074 | 0.3348228931427002 |
| Table Size (MB)         | 10.12             | 7.36                | 5.51               |
| Index Size (MB)         | 7.03              | 4.02                | 1.21               |

## FP32 Embeddings

- **Vectors:** 3633
- **Dimensions:** 384
- **Insert Time:** 7.52 seconds
- **Index Creation Time:** 0.54 seconds
- **Table Size:** 10.12 MB
- **Index Size:** 7.03 MB

## HALFVEC Embeddings

- **Vectors:** 3633
- **Dimensions:** 384
- **Insert Time:** 8.00 seconds
- **Index Creation Time:** 0.41 seconds
- **Table Size:** 7.36 MB
- **Index Size:** 4.02 MB

## BIT Embeddings

- **Vectors:** 3633
- **Dimensions:** 384
- **Insert Time:** 5.21 seconds
- **Index Creation Time:** 0.33 seconds
- **Table Size:** 5.51 MB
- **Index Size:** 1.21 MB
