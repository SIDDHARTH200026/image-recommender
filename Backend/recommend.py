import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_similar(features, query_idx, top_k=5):
    query_feat = features[query_idx:query_idx+1]
    similarities = cosine_similarity(query_feat, features)[0]
    top_indices = np.argsort(similarities)[-top_k-1:-1][::-1]
    return top_indices, similarities[top_indices]
