# Business Entity Resolution: v4 pipeline

Clean → shortlist (record → S1, 7 channels) → prefilter → 74 pair features → stage-1 LightGBM (A↔B cross-fit) →
fine-tuned cross-encoder (bge-reranker-v2-m3) → record matcher → collective stage 2 → decision (one S1 per record,
per-slice thresholds, zero-prediction rescue, cap 11).

**Train like test:** all train records use the same forward shortlist as test. Train S1 entities are split by entity
hash into H (2%, held-out evaluation), A (49%: stage-1 model A, cross-encoder, record matcher) and B (49%: stage-1
model B, stage 2 on a 10% slice).

## Reproduce (from `student_resource/`, data in `dataset/`)
CPU: 80 vCPU / 640 GB. GPU: one A100 40 GB (or ≥ 24 GB with bf16).
```bash
pip install -r requirements.txt
python src/normalize.py                                        # self-test: "normalize ok"
python src/er.py --stage testblock --data dataset --work work    # test cleaning + shortlist (can run in parallel)
python src/er.py --stage train     --data dataset --work work    # train shortlist, prefilter, features, stage 1
python src/ce.py export            --data dataset --work work    # cross-encoder training groups + train pairs to score
python src/ce.py train  --work work --model BAAI/bge-reranker-v2-m3 --out work/ce_model --groups-per-batch 16   # GPU
python src/ce.py score  --split train --work work --out work/ce_model --score-bs 1024                          # GPU
python src/er.py --stage test      --data dataset --work work --out output   # test prefilter, features, stage 1, candidate_pairs.tsv
python src/matcher.py              --data dataset --work work
python src/ce.py export_test       --data dataset --work work
python src/ce.py score  --split test  --work work --out work/ce_model --score-bs 1024                          # GPU
python src/stage2.py --stage fit   --data dataset --work work    # stage 2 + decision tuned on H half A, reported on H half B
python src/stage2.py --stage apply --data dataset --work work --out output   # output/matching_results.tsv
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```
Cloud automation used for the full run: `ops/cpu_single_chain.sh` (CPU VM) and `ops/gpu_chain.sh` (GPU VM).

## Files
| File | Purpose |
|---|---|
| `src/normalize.py` | Cleaning: anyascii, legal-form field, phones/e-mails/M/s/store numbers, OCR swaps, consonant + phonetic skeletons, unit codes, address ranges, Indian-script states, city aliases |
| `src/er.py` | Shortlist (5 channels + deep no-address + weak-name fallback + S1 reverse), prefilter, 74 features, stage-1 LightGBM, test scoring, candidate_pairs.tsv |
| `src/ce.py` | Cross-encoder: export groups (incl. synthetic pairs from split-A train records), fine-tune (pointwise + listwise), score (symmetric) |
| `src/matcher.py` | Record-to-record matcher (S2/S3), graph evidence for stage 2 |
| `src/stage2.py` | Collective stage 2 (sibling, source coverage, graph, cross-encoder), no-address model, decision layer, matching_results.tsv |
| `src/decode.py`, `src/dense.py`, `src/eda*`, `src/make_sample.py` | Earlier helpers (not used by the v4 run) |

Models: LightGBM (MIT), BAAI/bge-reranker-v2-m3 (Apache-2.0). Only the official train/test data; test is read-only.
