import time
import numpy as np

def find_similar_cases(query, model, index, metadata, top_k=5):
    """
    Find similar MPR cases using a preloaded FAISS index.

    Parameters:
    - query (str): New MPR issue text
    - model: Loaded SentenceTransformer model
    - index: Loaded FAISS index
    - metadata (list[dict]): Case metadata aligned with index
    - top_k (int): Number of similar cases to retrieve
    """

    # -----------------------------
    # Encode query
    # -----------------------------
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # -----------------------------
    # FAISS search
    # -----------------------------
    start_time = time.time()
    distances, indices = index.search(query_embedding, top_k)
    end_time = time.time()

    search_time_ms = round((end_time - start_time) * 1000, 2)
    print(f"FAISS retrieval time: {search_time_ms} ms")

    # -----------------------------
    # Confidence normalization
    # -----------------------------
    dists = distances[0]
    d_min = float(dists.min())
    d_max = float(dists.max())

    results = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        distance = float(dists[rank])

        # Normalize confidence per-query (scale-invariant)
        if d_max > d_min:
            confidence = 100 * (1 - (distance - d_min) / (d_max - d_min))
        else:
            confidence = 100.0  # edge case: all distances same

        confidence = round(confidence, 2)

        case = metadata[idx].copy()
        case["distance"] = round(distance, 4)
        case["confidence"] = confidence

        results.append(case)

    return results


