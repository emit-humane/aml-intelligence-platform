import json
with open("data/evaluation/evaluation_report.json") as f:
    r = json.load(f)

tm = r["transaction_metrics"]
cm = r.get("community_metrics", {})
pc = r.get("pattern_coverage", {})
print("=== SYSTEM 3 EVALUATION REPORT ===")
print(f"Generated: {r['generated_at']}")
print()
print("Transaction Metrics:")
print(f"  ROC-AUC:      {tm['roc_auc']:.4f}")
print(f"  PR-AUC:       {tm['pr_auc']:.4f}")
op = tm["operational"]
print(f"  @ thr=31:     P={op['precision']:.3f}  R={op['recall']:.3f}  F1={op['f1_fraud']:.3f}")
print(f"                TP={op['TP']}  FP={op['FP']}  FN={op['FN']}  TN={op['TN']}")
print(f"                FPR={op['false_positive_rate']:.3f}  FNR={op['false_negative_rate']:.3f}")
bf = tm.get("best_f1_metrics", {})
print(f"  Best F1:      F1={bf.get('f1_fraud',0):.3f}  P={bf.get('precision',0):.3f}  R={bf.get('recall',0):.3f}")
print()
print("Community/Ring Detection:")
print(f"  Rings total:           {cm.get('ring_ids_total','?')}")
print(f"  Rings >50% detected:   {cm.get('rings_50pct_detected','?')}")
print(f"  Full ring detect rate: {cm.get('full_ring_detection_rate',0):.4f}")
print(f"  Mean ring coverage:    {cm.get('mean_ring_coverage',0):.4f}")
print(f"  Partial detect rate:   {cm.get('partial_ring_detection_rate',0):.4f}")
print()
print("Pattern Coverage:")
print(f"  Overall detection rate: {pc.get('overall_detection_rate',0):.4f}")
for pat, v in pc.get("per_typology", {}).items():
    bar = "#" * int(v["detection_rate"] * 20)
    print(f"  {pat:25s}: {v['detected']:3d}/{v['total']:3d}  ({100*v['detection_rate']:5.1f}%)  {bar}")
print()
print("Benchmark Comparison:")
for bm, bv in r.get("benchmark_comparison", {}).items():
    our  = bv["our_roc_auc"]
    bm_  = bv["bm_auroc"]
    d    = bv.get("delta_auroc", our - bm_)
    sign = "+" if d >= 0 else ""
    f1d  = bv["our_best_f1"] - bv["bm_f1"]
    f1s  = "+" if f1d >= 0 else ""
    print(f"  vs {bm:10s}: AUROC {our:.4f} vs {bm_:.3f} ({sign}{d:.4f})  "
          f"F1 {bv['our_best_f1']:.3f} vs {bv['bm_f1']:.3f} ({f1s}{f1d:.3f})")
