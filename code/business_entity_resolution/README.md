# Business Entity Resolution: team matrix glitch, end-to-end reproduction

This folder regenerates both files in `output/`:

| File | md5 | Rows |
|---|---|---|
| `output/matching_results.tsv` | `e1b92115bbbba1560c336791347bc5c7` | 1,732,544 (the exact file uploaded to the leaderboard) |
| `output/candidate_pairs.tsv` | `ee4c2b80ce9ecdc4e996d8d2758c44f7` | 1,732,544; 70,449,807 candidate pairs (the blocking output of step 3) |

The official validator passes for the final file:
```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
PASS - no blocking issues found. Safe to submit.
```

## 0. Setup

- **Data:** the official `dataset/` (`train/`, `test/`) and `utils/validate_submission.py`. Nothing else is read: no external data, gazetteers or APIs.
- **Hardware:**
  - CPU steps: 64–80 vCPU and 512–640 GB RAM (we used GCP n2-highmem-64/80).
  - GPU steps: one H100 or A100 (bf16/fp16). The v6.2b cross-encoder was trained on an ml.g5.12xlarge (4× A10G).
  - Wall clock: about 7 h for v6.1, plus about 3 h for v6.2a / v6.2b, plus under 1 h of post-processing.
- **Install:** `pip install -r requirements.txt` (Python 3.12). The models BAAI/bge-m3 (MIT) and BAAI/bge-reranker-v2-m3 (Apache-2.0) download from the Hugging Face hub on first use and then run offline.
- **Paths:** in the commands below, `D=dataset`. Each stage writes into a work folder (`w4`, `w61`, …), and a stopped stage can be re-run on its own.
- **Run scripts:** the shell scripts in `src/v61_run_scripts/`, `src/v62a_selftraining/` and `src/v62b_france_crossencoder/` are the exact orchestration we ran, on several cloud machines. They copy files through our cloud buckets and contain `~/` paths, and AWS identifiers are redacted to `<AWS_ACCOUNT_ID>`. The commands in this README are the same steps without the cloud copies.
- **Hard-coded paths in one file:** `src/postprocess/fr_addback.py` uses `/home/Dell/...`. Change `/home/Dell` to your working folder (it needs `dataset/`, `w61/` and `w62a/`).

## 1. Source layout

| Folder | What it is |
|---|---|
| `src/v4/` | v4 pipeline: normalization, TF-IDF shortlist (5 channels + deep no-address, fallback and reverse searches), prefilter, 74 pair features, stage-1 LightGBM, cross-encoder fine-tune, record matcher, stage 2. Its outputs (candidates, prefilter, fine-tuned cross-encoder, label tables) are reused by v6.1. See `src/v4/README_v4.md`. |
| `src/v3/` | v3 pipeline. Its **France** rows (plus the look-alike filter) feed the agreement filter of the v6.2a pseudo-labels. See `README_v3.md`. |
| `src/v61/` | v6.1, the base model for all countries: `v61_gpu_embed.py` (bge-m3 top-20 search, GPU), `v61.py` (union with v4 candidates, prefilter, 77 features, stage 1, scoring, `candidate_pairs.tsv`), `v61_gpu_cos.py` (bge-m3 similarity features, GPU), `ce.py` (cross-encoder export and score), `matcher.py`, `stage2.py` (collective stage 2 and decision), `er.py` / `normalize.py` / `frstyle.py` (shared), `cmp_heldout.py` (paired held-out test) |
| `src/v61_run_scripts/` | exact v6.1 run order: GPU `g1` → CPU `runA3` / `runB3` / `runC3`, plus GPU `g2` → `runC` → GPU `g3` |
| `src/v62a_selftraining/` | `st_france.py`: one round of self-training on French test pairs (pseudo-labels), plus run scripts |
| `src/v62b_france_crossencoder/` | `fr_ce_groups.py` (French pseudo-labelled groups), `aws_train/` and `aws_score/` (cross-encoder fine-tune and scoring), `v62b_final.sh` |
| `src/postprocess/` | the final assembly (steps 7–10): `merge_by_country.py`, `france_lookalike_filter.py`, `france_rules.py`, `fr_addback.py`, `drop_ce_negative_addbacks.py`, `addback3_vm.py`, `e35x.py`, `noaddr_thr_vm.py` |
| `src/v61_heldout_test/` | the paired held-out test of v6.1 vs v4 |
| `src/analysis/` | every experiment and check behind the decisions: `friend_experiments/` (E1–E26), `v7_tests/` (self-training tests), `checks/` (leaderboard-calibration and file checks). None of it is needed to regenerate the outputs. |

## 2. v4 base (CPU + one GPU), from `src/v4/README_v4.md`
```bash
python src/v4/er.py --stage train --data $D --work w4          # shortlist, prefilter, features, stage 1 (see README_v4 for all stages)
python src/v4/ce.py ...                                         # fine-tune bge-reranker-v2-m3 on train pairs, then score
```
Kept for v6.1: `w4/{train,test}_pairs.parquet` (v4 candidates), the prefilter model, `v4_meta.json`, the fine-tuned cross-encoder, and `{train,test}_s1/_r.parquet` (normalized tables).

## 3. v6.1: all countries; produces `candidate_pairs.tsv` (`src/v61_run_scripts/` order)
```bash
# GPU: bge-m3 dense top-20 search per record, same country (g1.sh)
python src/v61/v61_gpu_embed.py --work w61 --emb emb --out dense
# CPU: union of v4 candidates + dense top-20, v4 prefilter
python src/v61/v61.py union --split train --data $D --work w61 --v4 w4 --prefilter <w4 prefilter model> --v4meta w4/v4_meta.json
python src/v61/v61.py union --split test  --data $D --work w61 --v4 w4 --prefilter <w4 prefilter model> --v4meta w4/v4_meta.json
# GPU: bge-m3 similarity features (g2.sh)
python src/v61/v61_gpu_cos.py --emb emb --work w61 --what train      # also --what test and --what copies
# CPU: 77 features (+ French-style copies of US training pairs) and stage 1 (reg x 3 seeds, A<->B cross-fit)
python src/v61/v61.py feats  --split train --what all --data $D --work w61 --v4meta w4/v4_meta.json
python src/v61/v61.py feats  --split test             --data $D --work w61 --v4meta w4/v4_meta.json
python src/v61/v61.py stage1 --part A --seeds 0,1,2 --data $D --work w61
python src/v61/v61.py stage1 --part B --seeds 0,1,2 --data $D --work w61
python src/v61/v61.py finish --data $D --work w61
python src/v61/v61.py score  --data $D --work w61 --out out61        # -> w61/test_scored.parquet, out61/candidate_pairs.tsv
# cross-encoder on the top-3 pairs (v4's fine-tuned bge-reranker-v2-m3; g3.sh), record matcher, stage 2, decision
python src/v61/ce.py export      --data $D --work w61
python src/v61/ce.py export_test --data $D --work w61
python src/v61/ce.py score --split train --work w61 --out <v4 ce_model> --score-bs 1024
python src/v61/ce.py score --split test  --work w61 --out <v4 ce_model> --score-bs 1024
python src/v61/matcher.py --data $D --work w61
python src/v61/stage2.py --stage fit   --data $D --work w61
python src/v61/stage2.py --stage apply --data $D --work w61 --out out61          # -> out61/matching_results.tsv
```
- `out61/candidate_pairs.tsv` = `output/candidate_pairs.tsv` (md5 `ee4c2b80…`).
- `out61/matching_results.tsv` has md5 `a8cff699667e5fc9213de28ae5feac8f`. We re-derived it byte for byte in step 10.

## 4. v6.2a: self-training on French test pairs (organiser-confirmed allowed)
```bash
# input: the 0.983297 France rows = v3 France + look-alike filter (src/v3 -> src/postprocess/france_lookalike_filter.py);
# st_france.py reads them as w61/v3_matching_results.tsv for its pseudo-label agreement filter
python src/v62a_selftraining/st_france.py --work w61 --data $D --half 0
python src/v62a_selftraining/st_france.py --work w61 --data $D --half 1
python src/v62a_selftraining/st_france.py --work w61 --merge            # -> w61/test_scored_v62a.parquet
# stage 2 + decision on the re-scored pairs: w62a = links to w61's files, with test_scored.parquet -> test_scored_v62a.parquet
python src/v61/stage2.py --stage apply --data $D --work w62a --out out62a        # md5 ec9e0022253815d88903288bb81ca866
```

## 5. France rows of file B (LB 0.986638)
```bash
python src/postprocess/merge_by_country.py --base out61/matching_results.tsv --part out62a/matching_results.tsv --countries France --out b0.tsv --test-dir $D/test
python src/postprocess/france_lookalike_filter.py --matching b0.tsv --out b1.tsv --test-dir $D/test --country France
python src/postprocess/france_rules.py --matching b1.tsv --out B.tsv --test-dir $D/test            # B md5 edca88869b7bd60aedbf86ff1964b35b
```
We rebuilt B byte for byte this way (`src/analysis/checks/runA.sh`, job 1).

## 6. v6.2b: cross-encoder adapted to France (pseudo-labelled French groups)
```bash
python src/v62b_france_crossencoder/fr_ce_groups.py --work w61 --data $D --out v62b      # 262,264 French groups + real US/India groups
python src/v62b_france_crossencoder/aws_train/ce.py train --work v62b --model BAAI/bge-reranker-v2-m3 --out v62b/model \
       --groups-per-batch 16 --max-steps 12000 --eval-every 12000 --ckpt-every 2000        # = aws_train/entry.py
python src/v61/ce.py export_test --data $D --work w62a                                    # v6.2a top-3 test pairs
python src/v62b_france_crossencoder/aws_score/ce.py score --split test --work w62a --out v62b/model --score-bs 1024   # -> v62b/*_test_ce.parquet
```

## 7. France add-back, then drop add-backs the France cross-encoder rejects (file G2, LB 0.987342)
```bash
python src/postprocess/fr_addback.py 0.7 add_0.7.tsv          # unmatched French records with v6.2a p1 >= 0.7 join their best S1 (cap 11)
python src/postprocess/france_lookalike_filter.py --matching add_0.7.tsv --out add_0.7_f.tsv --test-dir $D/test --country France
python src/postprocess/france_rules.py --matching add_0.7_f.tsv --out E_fr.tsv --test-dir $D/test
python src/postprocess/merge_by_country.py --base B.tsv --part E_fr.tsv --countries France --out G.tsv --test-dir $D/test     # md5 4478e903…
python src/postprocess/drop_ce_negative_addbacks.py --g G.tsv --d2 B.tsv --ce-dir v62b --test-dir $D/test --out G2.tsv        # md5 2c3680c9…
```
`--d2 B.tsv` is correct: the D2 file used on the leaderboard has B's France rows byte for byte.

## 8. More French adds gated by the France cross-encoder (file G4, LB 0.98743)
```bash
python src/postprocess/addback3_vm.py $D G2.tsv w61/test_scored_v62a.parquet v62b out       # -> out/G4.tsv, md5 05fdb1fe4262bb4339c2fd20d0f288be
```
- T2: p1 in [0.5, 0.7) and CE > 0.
- T3: p1 in [0.3, 0.5) and CE > 2.
- Rules for both: not an R1 look-alike, and a cap of 11 matches per S1.

## 9. Stricter no-address threshold on US/India (final file, LB 0.988)
```bash
# test frame with v6.1's pass-1 stage-2 probability for every US/India candidate (x_p2a); e35x.py also re-checks v6.1
python src/postprocess/e35x.py --data $D --work w61 --src src/v61 --out out_x --frame_out out_x/test_frame_usin.parquet
#   -> out_x/matching_results_v61_recomputed.tsv has md5 a8cff699… (= out61, byte for byte)
python src/postprocess/noaddr_thr_vm.py out/G4.tsv out_x/test_frame_usin.parquet final    # writes G6_na60..G6_na90
cp final/G6_na90.tsv output/matching_results.tsv                                            # md5 e1b92115bbbba1560c336791347bc5c7
cp out61/candidate_pairs.tsv output/candidate_pairs.tsv
```
This removes the G4 matches of US/India records without an address whose best stage-2 probability is in [0.50, 0.90): 27,511 matches from 26,115 S1. Nothing else changes.

## 10. Checks
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir $D/test --check-ids
md5sum output/*.tsv
```
