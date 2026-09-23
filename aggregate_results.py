"""
Walk one or more directories, find every *.console.log produced by the
run_*.sh scripts (ablation / sensitivity / seeds / efficiency / sampling
sweep), and extract into a single tidy CSV:

  - every hyperparameter from the run's argparse Namespace line
  - the final best Recall@K / NDCG@K
  - the average per-epoch training time and per-evaluation test time
    (useful for the efficiency analysis)

Usage:
    python scripts/aggregate_results.py History
    python scripts/aggregate_results.py History/ciao History/yelp
    python scripts/aggregate_results.py History -o my_summary.csv
"""

import argparse
import csv
import glob
import os
import re
import statistics as stats


NAMESPACE_RE = re.compile(r"Namespace\((.*)\)")
KV_RE = re.compile(r"(\w+)=('(?:[^'\\]|\\.)*'|[^,)]+)")
TRAIN_TIME_RE = re.compile(r"Training complete in ([\d.]+)s")
TEST_TIME_RE = re.compile(r"Testing complete in ([\d.]+)s")
BEST_RE = re.compile(
    r"Best\s+Recall@(\d+)\s+([\d.]+),\s*NDCG@(\d+)\s+([\d.]+)"
)

# Columns pulled out of the Namespace(...) line, in a sensible order.
NAMESPACE_FIELDS = [
    "dataset", "seed", "n_hid", "n_layers", "s_layers", "lr", "difflr",
    "reg", "steps", "noise_schedule", "noise_scale", "noise_min",
    "noise_max", "reweight", "sampling_steps", "sampling_noise",
    "eval_only", "checkpoint", "save_name", "model_dir",
]


def parse_namespace(text):
    m = NAMESPACE_RE.search(text)
    if not m:
        return {}
    kv = dict(KV_RE.findall(m.group(1)))
    return {k: v.strip().strip("'") for k, v in kv.items()}


def parse_log(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    row = {"log_path": path}
    row.update({k: parse_namespace(text).get(k, "") for k in NAMESPACE_FIELDS})

    train_times = [float(x) for x in TRAIN_TIME_RE.findall(text)]
    test_times = [float(x) for x in TEST_TIME_RE.findall(text)]
    row["avg_train_time_s"] = f"{stats.mean(train_times):.4f}" if train_times else ""
    row["avg_test_time_s"] = f"{stats.mean(test_times):.4f}" if test_times else ""
    row["n_train_epochs_timed"] = len(train_times)
    row["n_test_calls_timed"] = len(test_times)

    best_matches = BEST_RE.findall(text)
    if best_matches:
        topk, recall, _, ndcg = best_matches[-1]  # keep the final reported best
        row["topk"] = topk
        row["best_recall"] = recall
        row["best_ndcg"] = ndcg
    else:
        row["topk"] = row["best_recall"] = row["best_ndcg"] = ""

    return row


def find_logs(roots):
    logs = []
    for root in roots:
        if os.path.isfile(root) and root.endswith(".console.log"):
            logs.append(root)
        else:
            logs.extend(glob.glob(os.path.join(root, "**", "*.console.log"), recursive=True))
    return sorted(set(logs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="Directories (or specific .console.log files) to scan")
    ap.add_argument("-o", "--output", default="results_summary.csv", help="Output CSV path")
    args = ap.parse_args()

    logs = find_logs(args.roots)
    if not logs:
        print(f"No .console.log files found under: {args.roots}")
        return

    rows = [parse_log(p) for p in logs]
    fieldnames = ["log_path"] + NAMESPACE_FIELDS + [
        "topk", "best_recall", "best_ndcg",
        "avg_train_time_s", "n_train_epochs_timed",
        "avg_test_time_s", "n_test_calls_timed",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} log file(s) -> {args.output}")

    # Quick console preview: group by dataset and print mean/std of best_recall/ndcg
    # for rows where those values were found. Handy for the seed-robustness case.
    by_key = {}
    for r in rows:
        if not r["best_recall"]:
            continue
        key = (r.get("dataset", ""), os.path.dirname(r["log_path"]).split(os.sep)[-2]
               if os.sep in r["log_path"] else "")
        by_key.setdefault(r.get("dataset", "unknown"), []).append(
            (float(r["best_recall"]), float(r["best_ndcg"]))
        )

    if any(len(v) > 1 for v in by_key.values()):
        print("\nPer-dataset summary across matched runs (mean +/- std):")
        for dataset, vals in by_key.items():
            recalls = [v[0] for v in vals]
            ndcgs = [v[1] for v in vals]
            r_mean = stats.mean(recalls)
            n_mean = stats.mean(ndcgs)
            r_std = stats.stdev(recalls) if len(recalls) > 1 else 0.0
            n_std = stats.stdev(ndcgs) if len(ndcgs) > 1 else 0.0
            print(f"  {dataset:10s} n={len(vals):2d}  "
                  f"Recall={r_mean:.4f}+/-{r_std:.4f}  NDCG={n_mean:.4f}+/-{n_std:.4f}")


if __name__ == "__main__":
    main()

