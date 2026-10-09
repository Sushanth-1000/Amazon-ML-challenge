# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** matrix glitch
**Team Members:** Anirudha Rao, Tanay huddar, Sushanth C, Srinidhi B V
**Submission Date:** 29 September 2026 (final leaderboard file uploaded 27 September 2026)

---

## 1. Executive Summary

- **Pipeline (US and India):**
  - A dense bge-m3 search channel is added on top of a 7-channel TF-IDF shortlist.
  - A 77-feature LightGBM model scores candidates, then a fine-tuned bge-reranker-v2-m3 cross-encoder, then a collective stage 2.
  - Each Source 2/3 record is assigned to at most one Source 1 entity, with per-record-type thresholds tuned on held-out macro F0.5.
- **France (15% of test, no labels):**
  - The model adapts through self-training on French test pairs and a cross-encoder fine-tuned on French pseudo-labels.
  - A label-free "look-alike" filter removes French records that differ from a business by one swapped common word.
  - Missed French matches are re-added only when the France-adapted cross-encoder agrees.
- **Key finding:** test contains about twice as many non-matching "distractor" records per business as train. So borderline matches of records without an address are much less reliable on test than validation suggests, and a stricter no-address threshold improved the leaderboard.

---

## 2. Methodology

### 2.1 Problem Analysis

All numbers were measured on the full official data.

| | Train | Test |
|---|---|---|
| Source 1 entities | 2,206,821 | 1,732,544 (US, India, France) |
| Source 2+3 records | 10,320,219 | 9,969,589 |
| Records per S1 | 4.68 | **5.75** |
| Records that belong to some S1 | **74.0%** (ground truth) | **≈ 59%** (our final file matches 58.9%) |

- **One owner per record, no cross-country matches** (0 of 7.6M train pairs). Each record is a choice: its best S1, or none. Blocking runs per country, and files can be merged per country.
- **Singletons:** 5.6% of S1 have no match. They score 1 only with an empty prediction, so precision matters more than recall, as F0.5 intends.
- **Distractor records are near-clones of real businesses:**
  - The house number is shifted by 1–20 in 65% of US and 37% of India distractors, against 1.7% and 4.3% for true matches.
  - A business word is added ("group", "holdings", "overseas", …), with log-odds of about 8–10 for clone vs true.
- **Test has almost twice as many distractors per S1 as train:** about 2.3 vs 1.2 (records per S1 minus matches per S1). This shift was the root cause of several validation-vs-leaderboard gaps (Section 5).
- **Native scripts:** 9% of S2 and 5% of S3 names are Devanagari, Gujarati, Kannada, Tamil or Malayalam, while S1 is always Latin.
- **Records without an address** are the hardest type. The name alone must choose between chains and generic names; an identical name ranks the true S1 first only 61% of the time.
- **France is unseen:** no French labels exist, and a train-vs-test shift detector separates French from US text with AUC 1.000. French names carry legal forms and filler words (SARL, SAS, "services", "groupe", "fils", …), and decoys often swap one common word while keeping the street and number.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking (TF-IDF + dense retrieval) → gradient-boosted pair classifier → cross-encoder → collective stage 2 → per-record decision, plus label-free adaptation to the unseen country (self-training, a pseudo-label cross-encoder, and rules).

**Core Innovation:**
1. A multilingual dense retrieval channel (bge-m3) for cross-script and transliterated names.
2. For France without labels:
   - pseudo-label self-training;
   - a cross-encoder fine-tuned on 262,264 French pseudo-labelled groups;
   - a label-free look-alike rule.
3. Using **leaderboard differences between files that change only one country** as exact measurements. The metric is a per-S1 average and no match crosses countries, so this shows which changes help on the real test.

**Validation design:**
- 2% of train S1 (44,210, split "H" by an ID hash) are never trained on.
- Half A (22,160) tunes thresholds and models; half B (22,050) only scores.
- The metric is the official per-S1 macro F0.5 with singletons, with paired bootstrap 95% CIs (1,000 resamples).
- For France, "leave one country out" simulations stand in for an unseen country: US labelled with India unlabelled, and the reverse.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - v4 shortlist (record → S1, same country, cosine top-k with `sparse_dot_topn`, very frequent terms pruned), five TF-IDF channels:
    - name char 3-grams with spaces removed (top-20)
    - address words (top-20)
    - name + address word uni/bigrams (top-30)
    - phonetic name key, char 2–4-grams (top-20)
    - core name without legal form (top-20)
  - Plus a **deep search for records without an address** (name and phonetic channels, top-100), a **fallback** channel for records whose name finds nothing similar (no pruning, lower floor, deeper address search top-40), and a **reverse search** (S1 left without candidates searches the records, top-20).
  - **∪ bge-m3 dense top-20** per record on "name | address" (same country, fp16 top-k on GPU).
  - A LightGBM prefilter (v4's model and threshold).
- **Candidate pairs generated:** **70,449,807** test pairs, which is 40.7 per S1 or 7.1 per record (US 20,341,345 / India 35,455,944 / France 14,652,518). This is `output/candidate_pairs.tsv`. Every final match is inside it (validator PASS).
- **How true matches were not lost:**
  - Shortlist recall was measured on held-out at full S1 density: TF-IDF alone 0.9852, **with the dense channel 0.9929**.
  - The union kept 99.39% of true pairs, and the dense channel raised held-out F0.5 by +0.0026 [+0.0022, +0.0030].
  - Records without an address get a deeper name search.

---

## 4. Matching Model

**Features used (77):**
- **Name:** token-set / token-sort / partial ratios, space-free ratio and Jaro-Winkler, consonant-skeleton similarity (native scripts), core-name equality, name frequency.
- **Address:** token-set / sort ratios, number-token overlap, **first house number equality, absolute and relative shift, and a "small nonzero shift 1–20" flag**, postal-like codes equality.
- **Decoy / locality:** counts and **learned log-odds of added and missing name tokens** (cross-fitted label tables), locality tables.
- **Competition:** per-channel score, rank, gap to the record's best, margin between the top two, number of candidates, exact-name candidate count, uniqueness.
- **Dense:** bge-m3 cosine, gap to the record's best cosine, rank within the record.

**Model type:**
- **Stage 1:** LightGBM (MIT), with regularized settings (31 leaves, min_data_in_leaf 1000, feature_fraction 0.5, lambda_l2 10). It is cross-fitted between train halves A and B, averages 3 seeds, and also trains on "French-style" rewritten copies of US training pairs.
- **Cross-encoder:** BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M parameters), fine-tuned on training pairs, scoring each record's top-3 candidates.
- **Record matcher and collective stage 2:** LightGBM using sibling context, source coverage, record-graph edges and cross-encoder scores.
- **France only:**
  - **Self-training (v6.2a).** French test pairs with stage-1 p ≥ 0.9 and a margin ≥ 0.3 over the next candidate become pseudo-matches. Pairs with p ≤ 0.05, and the other candidates of pseudo-matched records, become pseudo-non-matches. Only pseudo-labels agreeing with an independent earlier France model are kept: 795,412 matches and 13.5M non-matches. Label tables are re-learned, and stage 1 is retrained with the pseudo rows (weight 0.5), cross-fitted over two halves of the French records.
  - **France-adapted cross-encoder (v6.2b).** bge-reranker-v2-m3 is fine-tuned on the real US/India groups plus 262,264 French pseudo-labelled groups (12,000 steps, pointwise + listwise loss).
- Every model is offline, MIT/Apache-2.0 and ≤ 8B parameters. No hosted APIs were used.

**Threshold selection method:**
- Each record goes to its highest-scoring S1, with a cap of 11 records per S1.
- It is kept if its pass-1 stage-2 probability clears a per-record-type threshold: no-address 0.50, native script 0.50, other 0.625, tuned for per-S1 macro F0.5 on held-out half A.
- The final file raises the **no-address threshold to 0.90 for US/India** (Section 5).
- **France post-processing, in order:**
  1. **R1 look-alike removal:** one swapped common word (a word in ≥ 100 French S1 names), same street and number, not a typo, abbreviation or filler word. Filler words are detected label-free as words placed after the legal form in ≥ 30% of cases.
  2. **Add-back:** unmatched French records with p1 ≥ 0.7 join their best S1.
  3. **Drop** add-backs the France-adapted cross-encoder scores < 0.
  4. **Extra tiers:** add p1 ∈ [0.5, 0.7) with CE > 0, and p1 ∈ [0.3, 0.5) with CE > 2.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), held-out validation (US + India, half B, 22,050 S1):** **0.9920** for the base model v6.1 (US 0.9925, India 0.9911). The previous version (v4) had 0.9897, and the paired gain is +0.00224 [+0.00181, +0.00273].
- **Public leaderboard (all countries), final file: 0.988.** Each row changes one country group, so the difference is that group's effect:

| File | Change | Public LB | Effect |
|---|---|---|---|
| v4 | all countries | 0.981643 | |
| v4 US/India + v3 France + look-alike filter | France | 0.983297 | |
| v6.1 US/India | US/India | 0.986041 | +0.00274 (validation predicted ≈ +0.0026) |
| + v6.2a self-trained France, filter, R1 | France | 0.986638 | +0.00060 |
| + French add-back p1 ≥ 0.7 | France | 0.986895* | +0.00034 (≈ 78% of adds true) |
| − add-backs with France-CE < 0 | France | 0.987342 | +0.00037 (the removed ones were ≈ 37% true) |
| + CE-gated tiers | France | 0.98743 | +0.00009 |
| **+ no-address threshold 0.50 → 0.90 (US/India)** | **US/India** | **0.988** | **≈ +0.0006** |

\*measured on a file that also carried a US/India re-ranker, which was worth −0.00008 by itself.

- **Common false positives (wrong merges):**
  - **Near-clone distractors:** same street, house number shifted by a few units, an added business word.
  - **French look-alikes with one swapped common word:** the removed R1 pairs were ≈ 91% wrong by the leaderboard.
  - **Borderline records without an address:** on test these are often distractors, because test has about twice as many distractors per business.
- **Common false negatives (missed matches):**
  - **French records the model was unsure of:** 3.3% of French records had stage-1 p ≥ 0.7 but stayed unassigned, vs 0.5% on held-out. Hence the France add-back, gated by the France cross-encoder.
  - **Before the dense channel:** India native-script and transliterated names never shortlisted (0.39 F0.5 points).
  - **Records without an address** ranked to the wrong same-name S1.
- **Validation vs leaderboard (lessons):**
  - Validation predicted whole-model changes well (v6.1: predicted +0.0026, measured +0.0027).
  - It did **not** predict changes aimed at low-confidence / no-address records:
    - a no-address re-ranker: validation +0.0005, leaderboard −0.00008;
    - a model adding matches for rejected records: validation +0.0018, **leaderboard −0.0058**, i.e. ≈ 27% of its adds were right, against 90% on validation.
  - Both follow from the distractor shift above. Going the other way (removing borderline no-address matches) gained on the leaderboard.

---

## 6. Conclusion

- Strong retrieval (TF-IDF + multilingual dense search) and a precise gradient-boosted matcher with a cross-encoder give 0.992 macro F0.5 on held-out US/India.
- The unseen country was handled without labels: self-training, a pseudo-label cross-encoder and a label-free look-alike rule.
- The main lesson: test differs from train in the share of non-matching records. So every change was checked on real test data through single-country leaderboard differences, and only changes that held up there were kept.

---

## Appendix

### A. Code Artefacts

- **Code:** `code/business_entity_resolution/`. `README.md` gives the exact commands, `requirements.txt` the pinned versions.
- **`src/`:**
  - `v4/`: base pipeline (shortlist, prefilter, cross-encoder fine-tune)
  - `v3/`: earlier model whose France rows seed the self-training agreement filter
  - `v61/`: main pipeline; produces `candidate_pairs.tsv`
  - `v61_run_scripts/`: exact run order
  - `v62a_selftraining/`: `st_france.py`
  - `v62b_france_crossencoder/`
  - `postprocess/`:
    - `merge_by_country.py`, `france_lookalike_filter.py`, `france_rules.py`
    - `fr_addback.py`, `drop_ce_negative_addbacks.py`, `addback3_vm.py`
    - `e35x.py` (US/India stage-2 frame), `noaddr_thr_vm.py` (final no-address threshold)
  - `analysis/`: all experiments and checks
- **Entry points:**
  - v6.1: `v61_gpu_embed.py` → `v61.py union / feats / stage1 / finish / score` (writes `candidate_pairs.tsv`) → `ce.py` → `matcher.py` → `stage2.py fit / apply`
  - France chain: README steps 4–8
  - Final no-address threshold: README step 9
- **Output checksums:**
  - `output/matching_results.tsv`: md5 `e1b92115bbbba1560c336791347bc5c7`
  - `output/candidate_pairs.tsv`: md5 `ee4c2b80ce9ecdc4e996d8d2758c44f7`

### B. Additional Results

**Paired held-out and simulation tests** (95% CIs; per-S1 macro F0.5 unless marked "pair-level"):

| Test | Result |
|---|---|
| v6.1 vs v4, held-out half B | all +0.00224 [+0.00181, +0.00273]; India +0.00555; US +0.00002 |
| bge-m3 search channel | +0.0026 [+0.0022, +0.0030]; shortlist recall 0.9852 → 0.9929 |
| bge-m3 similarity features, unseen country | +0.0025 [+0.0024, +0.0026] (pair-level) |
| French-style training copies, on French-style text | +0.0267 [+0.0265, +0.0268] (pair-level) |
| Self-training, US labelled → India unlabelled | +0.00899 [+0.00872, +0.00926]; a second round +0.01265 [+0.01235, +0.01295] |
| Self-training, India labelled → US unlabelled | +0.00096 [+0.00086, +0.00107] |
| Pseudo-label cross-encoder, India as unseen country | AP 0.9843 → 0.9910 (+0.0067 [+0.0061, +0.0073]) |
| French look-alike filter applied to US/India | −0.00007 (kept for France only) |

**Compliance.** These follow the organisers' answers to the challenge query form.
- **Uses of test data, all confirmed allowed:**
  - unsupervised statistics on test text (TF-IDF, token frequencies, blocking index)
  - self-training and pseudo-labels on test predictions
  - synthetic training copies from provided records
  - a small hand-written abbreviation list
- **Not used:** external data, gazetteers, postal/geographic libraries, hosted APIs.
- **Models:** BAAI/bge-m3 (MIT, ~568M), BAAI/bge-reranker-v2-m3 (Apache-2.0, ~568M), LightGBM (MIT); all offline.
- **Leaderboard use:** only to compare whole submission files.
