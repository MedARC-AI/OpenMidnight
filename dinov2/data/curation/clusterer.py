import h5py
import faiss
import argparse
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-file", type=str, default="tcga_embeddings.h5")
    parser.add_argument("--output-file", type=str, default="tcga_clusters.h5")
    parser.add_argument("--k1", type=int, default=10000)
    parser.add_argument("--k2", type=int, default=1000)
    return parser.parse_args()

def run_clustering(args):
    print("Loading embeddings...")
    with h5py.File(args.embed_file, 'r') as f:
        embeddings = f['embeddings'][:]

    N, dim = embeddings.shape
    print(f"Loaded {N} embeddings of dimension {dim}.")

    print(f"Training Level 1 K-Means (K1={args.k1})...")

    kmeans1 = faiss.Kmeans(d=dim, k=args.k1, niter=50, verbose=True, gpu=True)
    kmeans1.train(embeddings)

    print("Assigning patches to K1 clusters...")
    _, labels_k1 = kmeans1.index.search(embeddings, 1)
    labels_k1 = labels_k1.squeeze()

    print(f"Training Level 2 K-Means (K2={args.k2}) on K1 centroids...")
    kmeans2 = faiss.Kmeans(d=dim, k=args.k2, niter=50, verbose=True, gpu=True)
    kmeans2.train(kmeans1.centroids)

    print("Assigning K1 centroids to K2 clusters...")
    _, k1_to_k2_map = kmeans2.index.search(kmeans1.centroids, 1)
    k1_to_k2_map = k1_to_k2_map.squeeze()

    print("Mapping original patches to final K2 hierarchy...")
    labels_k2 = k1_to_k2_map[labels_k1]

    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset("cluster_k1", data=labels_k1)
        f.create_dataset("cluster_k2", data=labels_k2)

    print(f"Clustering complete. Saved to {args.output_file}.")

if __name__ == "__main__":
    run_clustering(get_args())
