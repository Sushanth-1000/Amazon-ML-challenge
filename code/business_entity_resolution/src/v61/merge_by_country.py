"""Merge two matching_results.tsv files by country: rows for --countries come from --part, all other rows from --base.

Safe because the metric is a per-Source-1 average and no match crosses countries (train ground truth has none;
records carry their own country). Refuses to write if a row would reference a record from another country.

  python scripts/merge_by_country.py --base friend_us_india.tsv --part my_france.tsv --countries France \
      --out output/matching_results.tsv
"""
import argparse
import csv
import sys


HEADERS = (["source1_entity_id", "matched_entity_ids"], ["source1_entity_id", "candidate_entity_ids"])


def read(path):
    """Returns (header, {s1_id: ids}); works for matching_results.tsv and candidate_pairs.tsv."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    assert rows[0] in HEADERS, f"{path}: bad header {rows[0]}"
    return rows[0], {r[0]: (r[1] if len(r) > 1 else "") for r in rows[1:]}


def countries(path):
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        head = next(rd)
        i, c = head.index("entity_id"), head.index("country")
        return {r[i]: r[c] for r in rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-dir", default="dataset/test")
    a = ap.parse_args()

    s1 = countries(f"{a.test_dir}/test_source1.tsv")
    rec = {**countries(f"{a.test_dir}/test_source2.tsv"), **countries(f"{a.test_dir}/test_source3.tsv")}
    (head, base), (head_p, part) = read(a.base), read(a.part)
    assert head == head_p, f"--base and --part are different file kinds: {head} vs {head_p}"
    missing = [i for i in s1 if i not in base or (s1[i] in a.countries and i not in part)]
    if missing:
        sys.exit(f"{len(missing)} Source-1 ids missing, e.g. {missing[:3]}")

    out, taken = [], 0
    for i, c in s1.items():
        ids = part[i] if c in a.countries else base[i]
        taken += c in a.countries
        cross = [x for x in ids.split(",") if x and rec.get(x) != c]
        if cross:
            sys.exit(f"{i} ({c}) lists records from another country or unknown ids: {cross[:3]}")
        out.append((i, ids))

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(head) + "\n")
        f.writelines(f"{i}\t{ids}\n" for i, ids in out)
    print(f"wrote {a.out}: {len(out):,} rows, {taken:,} from --part ({', '.join(a.countries)}), "
          f"{len(out) - taken:,} from --base")


if __name__ == "__main__":
    main()
