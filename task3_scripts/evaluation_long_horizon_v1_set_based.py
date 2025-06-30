import os
import json
import numpy as np
import hnswlib
from tqdm import tqdm
import multiprocessing
import argparse
from collections import defaultdict

# Default paths
CATALOG_EMB    = "cache/catalog_embeddings.npy"
CATALOG_TITLES = "cache/catalog_titles.txt"
PRED_EMB       = "cache/gemma2-9b_test_long_horizon_embeddings.npy"
PRED_TITLES    = "cache/gemma2-9b_test_long_horizon_titles.txt"
PRED_JSONL     = "long_horizon_experiments/gemma2-9b-test_full_long_horizon_parsed.jsonl"
INDEX_PATH     = "cache/catalog_hnsw_index.bin"
OUTPUT_JSONL   = "gemma2-9b-long_horizon_topk_user_predictions_hnsw_userlevel.jsonl"

# Default settings
DEFAULT_THRESH = 0.9
KS = [10, 20, 50, 100]
NUM_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 50_000

def parse_args():
    parser = argparse.ArgumentParser("Evaluate HNSW predictions on Amazon data.")

    # Command-line arguments for all paths
    parser.add_argument('--catalog-emb', type=str, default=CATALOG_EMB,
                        help='Path to catalog embeddings .npy file')
    parser.add_argument('--catalog-titles', type=str, default=CATALOG_TITLES,
                        help='Path to catalog titles .txt file')
    parser.add_argument('--pred-emb', type=str, default=PRED_EMB,
                        help='Path to predicted embeddings .npy file')
    parser.add_argument('--pred-titles', type=str, default=PRED_TITLES,
                        help='Path to predicted titles .txt file')
    parser.add_argument('--predictions', type=str, default=PRED_JSONL,
                        help='Path to predicted queries JSONL file')
    parser.add_argument('--index-path', type=str, default=INDEX_PATH,
                        help='Path to save/load HNSW index')
    parser.add_argument('--output', type=str, default=OUTPUT_JSONL,
                        help='Path to write output JSONL with top-k user predictions')
    parser.add_argument('-f', '--force-rebuild', action='store_true',
                        help='Force rebuild the HNSW index')
    parser.add_argument('--queries-per-user', type=int, default=10,
                        help='Number of predicted queries per user')
    parser.add_argument('--sim-threshold', type=float, default=DEFAULT_THRESH,
                        help='Similarity threshold for considering an item relevant')

    return parser.parse_args()

def load_embeddings(emb_path, titles_path):
    emb = np.load(emb_path).astype('float32')
    titles = [l.strip() for l in open(titles_path)]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, titles

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in tqdm(f, desc="Loading JSONL"):
            data.append(json.loads(line))
    return data

def build_or_load_index(emb, index_path, force=False):
    dim, max_e = emb.shape[1], emb.shape[0]
    if not force and os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        idx = hnswlib.Index(space='cosine', dim=dim)
        idx.load_index(index_path)
    else:
        print("Building new HNSW index...")
        idx = hnswlib.Index(space='cosine', dim=dim)
        idx.init_index(max_elements=max_e, ef_construction=300, M=48)
        idx.set_num_threads(NUM_THREADS)
        for start in tqdm(range(0, max_e, BATCH_SIZE), desc="Indexing batches"):
            end = min(max_e, start + BATCH_SIZE)
            idx.add_items(emb[start:end], np.arange(start, end, dtype=np.int32))
        idx.save_index(index_path)
    idx.set_num_threads(NUM_THREADS)
    idx.set_ef(max(KS))
    return idx

def evaluate_user_metrics(user_entries, catalog_titles, catalog_map, sim_thresh):
    sums = {k: {'recall':0., 'precision':0., 'ndcg':0.} for k in KS}
    counts = {k: 0 for k in KS}

    for user_id, true_titles, merged_neighbors in tqdm(user_entries, desc="Evaluating user metrics"):
        if not true_titles or not any(t in catalog_map for t in true_titles):
            continue
        valid_trues = [t for t in true_titles if t in catalog_map]
        if not valid_trues:
            continue
        for K in KS:
            topk_titles = merged_neighbors.get(K, [])
            if not topk_titles:
                continue
            retrieved = [t for t in topk_titles if t in catalog_map]
            if not retrieved:
                continue
            rel = np.zeros(K)
            for i, title in enumerate(retrieved[:K]):
                if title in catalog_map:
                    max_sim = max(np.dot(catalog_map[title], catalog_map[t])
                                  for t in valid_trues if t in catalog_map)
                    rel[i] = 1.0 if max_sim >= sim_thresh else 0.0
            recall = rel.sum() / len(valid_trues)
            precision = rel.sum() / K
            dcg = sum((rel[i] / np.log2(i + 2)) for i in range(min(K, len(rel))))
            idcg = sum((1.0 / np.log2(i + 2)) for i in range(min(len(valid_trues), K)))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            sums[K]['recall'] += recall
            sums[K]['precision'] += precision
            sums[K]['ndcg'] += ndcg
            counts[K] += 1
    return sums, counts

def process_batch_queries(user_batch, pred_map, idx, catalog_titles):
    all_queries = []
    query_map = {}
    for user_idx, user_entry in enumerate(user_batch):
        queries = user_entry['queries']
        for q_idx, query in enumerate(queries):
            if query.strip() in pred_map:
                query_map[len(all_queries)] = (user_idx, q_idx)
                all_queries.append(query.strip())
    if not all_queries:
        return []
    q_embs = np.stack([pred_map[q] for q in all_queries])
    max_k = max(KS)
    neighbors_per_query = {K: max(1, K // 10) for K in KS}
    max_neighbors = max(neighbors_per_query.values())
    labels_batch, _ = idx.knn_query(q_embs, k=max_neighbors, num_threads=NUM_THREADS)
    user_neighbors = defaultdict(lambda: defaultdict(list))
    for q_idx, labels in enumerate(labels_batch):
        if q_idx in query_map:
            user_idx, _ = query_map[q_idx]
            for K in KS:
                k_per_q = neighbors_per_query[K]
                for j in labels[:k_per_q]:
                    title = catalog_titles[int(j)]
                    user_neighbors[user_idx][K].append(title)
    results = []
    for user_idx, k_neighbors in user_neighbors.items():
        user_entry = user_batch[user_idx]
        merged_neighbors = {}
        for K in KS:
            merged_neighbors[K] = k_neighbors[K][:K]
        results.append((user_entry['user_id'], user_entry['true_titles'], merged_neighbors))
    return results

def main():
    args = parse_args()
    print("Starting evaluation with args:", args)
    print(f'Using {NUM_THREADS} threads for HNSW indexing and querying.')
    print(f"Using {BATCH_SIZE} batch size for processing.")
    QUERY_PER_USER = args.queries_per_user

    # Load data from args
    catalog_emb, catalog_titles = load_embeddings(args.catalog_emb, args.catalog_titles)
    print(f"Catalog: {len(catalog_titles)} titles, {catalog_emb.shape[0]} embeddings")
    pred_emb, pred_titles = load_embeddings(args.pred_emb, args.pred_titles)
    print("Loaded embeddings and titles.")
    print(f"Predictions: {len(pred_titles)} titles, {pred_emb.shape[0]} embeddings")

    preds = load_jsonl(args.predictions)
    print(f"Loaded {len(preds)} prediction entries.")

    catalog_map = dict(zip(catalog_titles, catalog_emb))
    pred_map = dict(zip(pred_titles, pred_emb))
    print("Building HNSW index...")
    idx = build_or_load_index(catalog_emb, index_path=args.index_path, force=args.force_rebuild)

    valid = []
    for entry in tqdm(preds, desc="Filtering entries"):
        true_titles = entry.get('true_history_titles', [])
        pred_queries_list = entry.get('predicted_history_titles', [])
        if (len(true_titles) > 0 and
            len(pred_queries_list) == 1 and
            len(pred_queries_list[0]) == QUERY_PER_USER):
            valid.append({
                'user_id': entry['user_id'],
                'true_titles': true_titles,
                'queries': pred_queries_list[0]
            })
    print(f"Valid entries: {len(valid)}/{len(preds)}")

    user_results = []
    for batch_start in tqdm(range(0, len(valid), BATCH_SIZE), desc="Processing user batches"):
        batch = valid[batch_start:min(batch_start + BATCH_SIZE, len(valid))]
        batch_results = process_batch_queries(batch, pred_map, idx, catalog_titles)
        user_results.extend(batch_results)

    print(f"Processed {len(user_results)} users with valid results")

    sums, counts = evaluate_user_metrics(user_results, catalog_titles, catalog_map, args.sim_threshold)

    for K in KS:
        if counts[K] > 0:
            print(f"K={K} Recall@{K}: {sums[K]['recall']/counts[K]:.6f} "
                  f"Precision@{K}: {sums[K]['precision']/counts[K]:.6f} "
                  f"NDCG@{K}: {sums[K]['ndcg']/counts[K]:.6f}")
        else:
            print(f"K={K}: no valid cases.")

    with open(args.output, 'w') as fout:
        for i, (user_id, true_titles, merged_neighbors) in enumerate(user_results[:10]):
            out = {
                'user_id': user_id,
                'true_history': true_titles,
                'predicted_queries': valid[i]['queries'] if i < len(valid) else [],
                'retrieved_topk': merged_neighbors
            }
            fout.write(json.dumps(out) + "\n")

    print("Done.")

if __name__ == '__main__':
    main()
