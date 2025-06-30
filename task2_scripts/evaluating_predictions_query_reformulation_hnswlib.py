import os
import json
import numpy as np
import hnswlib
from tqdm import tqdm
import multiprocessing
import argparse
from collections import defaultdict

# Default paths (overridable via command-line arguments)
DEFAULT_CATALOG_EMB    = "cache/catalog_embeddings.npy"
DEFAULT_CATALOG_TITLES = "cache/catalog_titles.txt"
DEFAULT_PRED_EMB       = "cache/embeddings.npy"
DEFAULT_PRED_TITLES    = "cache/titles.txt"
DEFAULT_PRED_JSONL     = "llm_experiments/full_parsed.jsonl"
DEFAULT_INDEX_PATH     = "cache/catalog_hnsw_index.bin"
DEFAULT_OUTPUT_JSONL   = "full_topk_user_predictions_hnsw.jsonl"

# Default settings
DEFAULT_THRESH = 0.9
KS = [10, 20, 50, 100]
NUM_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 50_000

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HNSW-based top-K retrieval on Amazon data with customizable paths and settings."
    )
    parser.add_argument(
        '-f', '--force-rebuild', action='store_true',
        help='Force rebuild the HNSW index instead of loading an existing one.'
    )
    parser.add_argument(
        '--queries-per-true', type=int, default=10,
        help='Number of predicted queries per true title to use for retrieval (default: 10).'
    )
    parser.add_argument(
        '--sim-threshold', type=float, default=DEFAULT_THRESH,
        help=f'Cosine similarity threshold to consider a prediction relevant (default: {DEFAULT_THRESH}).'
    )
    parser.add_argument(
        '--output', type=str, default=DEFAULT_OUTPUT_JSONL,
        help='Output JSONL file to save sample user predictions and retrieved top-K results.'
    )
    parser.add_argument(
        '--predictions', type=str, default=DEFAULT_PRED_JSONL,
        help='Input JSONL file containing model predictions to evaluate.'
    )
    parser.add_argument(
        '--catalog-emb', type=str, default=DEFAULT_CATALOG_EMB,
        help='Path to catalog embeddings file (.npy) used to build the index.'
    )
    parser.add_argument(
        '--catalog-titles', type=str, default=DEFAULT_CATALOG_TITLES,
        help='Path to catalog titles file (.txt), one title per line.'
    )
    parser.add_argument(
        '--pred-emb', type=str, default=DEFAULT_PRED_EMB,
        help='Path to predicted query/title embeddings file (.npy).'
    )
    parser.add_argument(
        '--pred-titles', type=str, default=DEFAULT_PRED_TITLES,
        help='Path to predicted query/title texts file (.txt), one title per line.'
    )
    parser.add_argument(
        '--index-path', type=str, default=DEFAULT_INDEX_PATH,
        help='Path to save or load the built HNSW index file (.bin).'
    )
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

def evaluate_metrics_batch(all_labels, catalog_titles, catalog_map, sim_thresh, queries_per_true):
    """
    all_labels: list of ((entry_idx, true_title, q_idx), neighbor_indices)
    returns: sums, counts dicts for each K
    """
    sums = {k: {'recall':0., 'precision':0., 'ndcg':0.} for k in KS}
    counts = {k: 0 for k in KS}

    labels_by_task = defaultdict(list)
    for (entry_idx, true_title, _), labels in all_labels:
        labels_by_task[(entry_idx, true_title)].append(labels)

    for (entry_idx, true_title), label_lists in tqdm(labels_by_task.items(), desc="Evaluating metrics"):
        if true_title not in catalog_map:
            continue
        t_emb = catalog_map[true_title]
        for K in KS:
            n_per_q = max(1, K // queries_per_true)
            merged = []
            for lbls in label_lists:
                merged.extend([catalog_titles[int(idx)] for idx in lbls[:n_per_q]])
            merged = merged[:K]

            embs = [catalog_map[t] for t in merged if t in catalog_map]
            if not embs:
                recall = precision = ndcg = 0.0
            else:
                sims = np.dot(np.stack(embs), t_emb)
                rel = sims >= sim_thresh
                recall = 1.0 if rel.any() else 0.0
                precision = rel.sum() / float(K)
                dcg = sum((1.0 if rel[i] else 0.0) / np.log2(i + 2) for i in range(len(rel)))
                n_rel = int(rel.sum())
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel, K)))
                ndcg = (dcg / idcg) if idcg > 0 else 0.0

            sums[K]['recall']    += recall
            sums[K]['precision'] += precision
            sums[K]['ndcg']      += ndcg
            counts[K] += 1

    return sums, counts


def main():
    args = parse_args()
    print("Starting evaluation with args:", args)
    print(f'Using {NUM_THREADS} threads for HNSW indexing and querying.')
    print(f"Using {BATCH_SIZE} batch size for processing.")

    # Load data
    catalog_emb, catalog_titles = load_embeddings(args.catalog_emb, args.catalog_titles)
    pred_emb, pred_titles = load_embeddings(args.pred_emb, args.pred_titles)
    print("Loaded embeddings and titles.")
    print(f"Catalog: {len(catalog_titles)} titles, {catalog_emb.shape[0]} embeddings")
    print(f"Predictions: {len(pred_titles)} titles, {pred_emb.shape[0]} embeddings")

    preds = load_jsonl(args.predictions)
    print(f"Loaded {len(preds)} prediction entries.")

    catalog_map = dict(zip(catalog_titles, catalog_emb))
    pred_map = dict(zip(pred_titles, pred_emb))
    print("Building HNSW index...")
    idx = build_or_load_index(catalog_emb, args.index_path, force=args.force_rebuild)

    valid = []
    for entry in tqdm(preds, desc="Filtering entries"):
        trues = entry.get('true_history_titles', [])
        preds_list = entry.get('predicted_history_titles', [])
        if len(trues) > 0 and len(preds_list) == len(trues) and all(len(preds_list[i]) >= args.queries_per_true for i in range(len(trues))):
            valid.append({'user_id': entry['user_id'], 'trues': trues, 'preds': preds_list})
    print(f"Valid entries: {len(valid)}/{len(preds)}")

    total_sums = {k: {'recall':0., 'precision':0., 'ndcg':0.} for k in KS}
    total_counts = {k: 0 for k in KS}

    for bstart in tqdm(range(0, len(valid), BATCH_SIZE), desc="Processing batches"):
        batch = valid[bstart:bstart + BATCH_SIZE]
        qp = args.queries_per_true
        tasks, q_embs = [], []
        for i, entry in enumerate(batch):
            for true_title, q_list in zip(entry['trues'], entry['preds']):
                for q_idx, q in enumerate(q_list[:qp]):
                    q_clean = q.strip()
                    if q_clean in pred_map:
                        tasks.append((i + bstart, true_title, q_idx))
                        q_embs.append(pred_map[q_clean])
        if not q_embs:
            continue
        q_embs = np.stack(q_embs)

        all_labels = []
        for K in KS:
            print(f"Processing K={K} for batch {bstart}")
            n_per_q = max(1, K // qp)
            labels_batch, _ = idx.knn_query(q_embs, k=n_per_q, num_threads=NUM_THREADS)
            all_labels.extend([((ei, tt, qi), lbls) for (ei, tt, qi), lbls in zip(tasks, labels_batch)])

        sums, counts = evaluate_metrics_batch(all_labels, catalog_titles, catalog_map, args.sim_threshold, qp)
        for K in KS:
            total_sums[K]['recall']    += sums[K]['recall']
            total_sums[K]['precision'] += sums[K]['precision']
            total_sums[K]['ndcg']      += sums[K]['ndcg']
            total_counts[K]            += counts[K]

    for K in KS:
        if total_counts[K] > 0:
            print(f"K={K} Recall@{K}: {total_sums[K]['recall']/total_counts[K]:.6f} "
                  f"Precision@{K}: {total_sums[K]['precision']/total_counts[K]:.6f} "
                  f"NDCG@{K}: {total_sums[K]['ndcg']/total_counts[K]:.6f}")
        else:
            print(f"K={K}: no valid cases.")

    with open(args.output, 'w') as fout:
        for entry in valid[:10]:
            out = {'user_id': entry['user_id'], 'true_history': entry['trues'], 'predicted_history': entry['preds'], 'retrieved_topk': {}}
            qp = args.queries_per_true
            q_embs = np.stack([pred_map[q.strip()] for sub in entry['preds'][:qp] for q in sub if q.strip() in pred_map])
            for K in KS:
                n_per_q = max(1, K // qp)
                labels_batch, _ = idx.knn_query(q_embs, k=n_per_q)
                merged = []
                for row in labels_batch:
                    for j in row:
                        merged.append(catalog_titles[int(j)])
                out['retrieved_topk'][K] = merged[:K]
            fout.write(json.dumps(out) + "\n")

    print("Done.")

if __name__ == '__main__':
    main()

