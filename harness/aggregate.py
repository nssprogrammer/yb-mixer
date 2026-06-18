"""Collect runs/*.json into a results table. Usage: python aggregate.py [runs_dir]"""
import json, glob, os, sys, collections

d = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    rows.append(json.load(open(f)))
if not rows:
    print("no runs found in", d); raise SystemExit

by_task = collections.defaultdict(list)
for r in rows:
    by_task[r["task"]].append(r)

print(f"\n{'task':<8} {'model':<12} {'params':>10} {'best_val_acc':>13} {'epochs':>7} {'time(s)':>9}")
print("-" * 64)
for task in sorted(by_task):
    for r in sorted(by_task[task], key=lambda x: -x["best_val_acc"]):
        print(f"{task:<8} {r['model']:<12} {r['params']:>10,} "
              f"{r['best_val_acc']:>13.4f} {r['epochs']:>7} {r['seconds']:>9.0f}")
    print("-" * 64)
