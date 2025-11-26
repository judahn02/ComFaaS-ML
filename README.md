# ComFaaS-ML
ComFaaS-ML is a latency-aware serverless scheduler using a compute regressor, Top-K pruner, and listwise/pairwise reranker to pick near-optimal nodes per request. It operates on decision-time features only, avoids label leakage, and scales to large clusters with fast per-request inference.

## This is how to run the simulations
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

