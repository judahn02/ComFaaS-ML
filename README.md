# ComFaaS-ML



Learning-based scheduler for Function-as-a-Service environments. Operates only on decision-time features, predicts compute latency, prunes the search space, reranks candidates, and selects near-optimal nodes under heterogeneous load.

---
. **[Overview](#overview)** .
**[Pipeline](#pipeline)** .
**[Reproducibility](#reproducibility)** .

---

## Overview

ComFaaS-ML processes synthetic or real request streams. Each row represents a (req_id, node_id) pair with context, resource state, network signals, and decision-time estimates. The pipeline builds a compact feature frame, trains a compute regressor, prunes to a viable candidate set, and applies a listwise/pairwise reranker to produce a final top-1 node per request.

## Pipeline

1. Generate or load dataset.
2. Chronological split.
3. Preprocess: feasibility mask → decision-time estimates → safe feature subset + aligned labels.
4. Train regressor.
5. Predict compute + total estimates.
6. Prune with top-K selection.
7. Align labels to pruned set.
8. Train reranker.
9. Produce final per-request ranking.
10. Compare with heuristic baseline.

## Reproducibility
```
python ComFaaS_ML_80.1.2.py \
  --mode hybrid \
  --reqs 20000 --nodes 20 --seed 1337 \
  --K 16 --lambda_tail 0.7 --eps 0.15 \
  --train_reranker --fast_reranker \
  --rerank_epochs 16 --rerank_batch 32768 \
  --listwise_w 1.0 --pairwise_w 0.0 \
  --save_parquet \
  --outdir 80.1.2_hybridml_S1_fast
```

```
python ComFaaS_ML_80.1.2.py \
  --mode pure-ml \
  --reqs 20000 --nodes 20 --seed 1337 \
  --K 16 --lambda_tail 0.7 --eps 0.15 \
  --train_reranker --fast_reranker \
  --rerank_epochs 16 --rerank_batch 32768 \
  --listwise_w 1.0 --pairwise_w 0.0 \
  --save_parquet \
  --outdir 80.1.2_pureml_S1_fast
```

