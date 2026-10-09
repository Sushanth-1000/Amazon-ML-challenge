# Business Entity Resolution: v3 pipeline

Normalize → block (TF-IDF top-k, per country, record → S1) → prefilter → LightGBM stage 1 (2 folds) →
fine-tuned cross-encoder (bge-reranker-v2-m3) on uncertain pairs → stage 2 (sibling context + cross-encoder) →
assign each S2/S3 record to at most one S1.

## Reproduce
Run from the `student_resource/` directory (data in `dataset/`). CPU steps: 64 vCPU / 512 GB recommended.
GPU steps (`ce.py train|score`): one A100 40 GB (or any ≥ 24 GB GPU with bf16).

```bash
pip install -r requirements.txt
python src/normalize.py                                   # self-test, prints "normalize ok"
python src/er.py --stage train     --data dataset --work work --reverse-k 40
python src/decode.py               --data dataset --work work
python src/er.py --stage testblock --data dataset --work work   # optional: can run in parallel with train
python src/er.py --stage test      --data dataset --work work --out output
python src/ce.py export --split train --data dataset --work work
python src/ce.py train  --work work --model BAAI/bge-reranker-v2-m3 --out work/ce_model --bs 128 --lr 2e-5 --max-len 96 --eval-every 2000
python src/ce.py export --split val  --data dataset --work work
python src/ce.py export --split test --data dataset --work work
python src/ce.py score  --split val  --work work --model work/ce_model --score-bs 1024 --max-len 96
python src/ce.py score  --split test --work work --model work/ce_model --score-bs 1024 --max-len 96
python src/stage2.py               --data dataset --work work
python src/er.py --stage final     --data dataset --work work --out output
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```
Outputs: `output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv` (exactly the pairs the model scores,
i.e. the prefilter survivors). Intermediate results are cached in `work/`, so a stopped run resumes.
Measured v3 run: ~3.5 h CPU (n2-highmem-64) + ~1 h GPU (A100 spot).

## Files
| File | Purpose |
|---|---|
| `src/normalize.py` | Name/address cleaning (anyascii, legal-form field, Indian-script states, city aliases, OCR swaps, phones), consonant skeleton |
| `src/er.py` | Blocking, ranking features, prefilter, pair features, 2-fold LightGBM, test scoring, final decision + outputs |
| `src/decode.py` | Decision-rule tuning (threshold / margin / expected-F0.5) on validation half A, report on half B |
| `src/ce.py` | Cross-encoder: export pairs, fine-tune, score (GPU) |
| `src/stage2.py` | Stage-2 re-scoring with sibling context + cross-encoder features, entity-grouped cross-fit |
| `src/eda/`, `src/eda.py`, `src/make_sample.py` | v2 measurements that set the blocking design; sampling helper |

Models: LightGBM (MIT), BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M). No external data or lookups.
