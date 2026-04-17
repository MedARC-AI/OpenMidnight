import h5py
import argparse
import numpy as np
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-file", type=str, default="tcga_embeddings.h5")
    parser.add_argument("--cluster-file", type=str, default="tcga_clusters.h5")
    parser.add_argument("--output-txt", type=str, default="curated_dataset.txt")
    parser.add_argument("--samples-per-cluster", type=int, default=500)
    return parser.parse_args()

def sample_dataset(args):
    with h5py.File(args.embed_file, 'r') as f_embed, h5py.File(args.cluster_file, 'r') as f_cluster:
        metadata = f_embed['metadata'][:]
        labels_k2 = f_cluster['cluster_k2'][:]

    cluster_to_indices = {}
    for i, cluster_id in enumerate(labels_k2):
        if cluster_id not in cluster_to_indices:
            cluster_to_indices[cluster_id] = []
        cluster_to_indices[cluster_id].append(i)

    selected_indices =[]
    for cluster_id, indices in cluster_to_indices.items():
        if len(indices) > args.samples_per_cluster:
            sampled = np.random.choice(indices, size=args.samples_per_cluster, replace=False)
        else:
            sampled = indices
        selected_indices.extend(sampled)

    np.random.shuffle(selected_indices)

    print(f"Selected {len(selected_indices)} patches.")
    with open(args.output_txt, 'w') as f:
        for idx in selected_indices:
            meta_str = metadata[idx].decode('utf-8')
            f.write(f"{meta_str}\n")

    print(f"Saved curated list to {args.output_txt}.")

if __name__ == "__main__":
    sample_dataset(get_args())
