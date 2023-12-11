import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import symnmf

def initialize_H(W, k):
    m = np.mean(W)
    return np.random.uniform(0, 2 * np.sqrt(m / k), (len(W), k))

def read_data(filename):
    with open(filename, 'r') as file:
        data = [list(map(float, line.strip().split(','))) for line in file]
    return np.array(data)

def apply_symnmf(W, H, n, k):
    """Apply SymNMF algorithm on the dataset and return labels."""
    h_matrix = symnmf.symnmf(W, H, n, k)
    labels = np.argmax(h_matrix, axis=1)
    return labels

def apply_kmeans(data, k):
    """Apply KMeans algorithm on the dataset and return labels."""
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300).fit(data)
    return kmeans.labels_

def compute_silhouette(data, labels):
    """Compute silhouette score for the clustering results."""
    return silhouette_score(data, labels)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 analysis.py <k> <input_file.txt>")
        sys.exit(1)
    
    k = int(sys.argv[1])
    filename = sys.argv[2]

    # Reading data
    X = read_data(filename)
    n, d = X.shape  # Get dimensions of X
    X = X.tolist()
    
    W = symnmf.norm(X, n, d)        
    # Initialize H
    H = initialize_H(W, k)
    H = H.tolist()
    # Get final H using the C extension
    H = symnmf.symnmf(W, H, n, k)

    print(H)

    
    symnmf_score = compute_silhouette(X, symnmf_labels)
    
    # KMeans clustering
    kmeans_labels = apply_kmeans(X, k)
    kmeans_score = compute_silhouette(X, kmeans_labels)
    
    print(f"nmf: {symnmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")
