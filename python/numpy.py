import numpy as np

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









