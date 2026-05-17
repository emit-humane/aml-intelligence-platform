"""
External benchmark constants.

From: "Multi-GNN AML Detection" paper and Tide LI/HI baseline.
These serve as the comparison baseline for evaluation_report.json.
"""

BENCHMARKS = {
    "Tide_LI": {
        "description": "Tide Low-Involvement baseline (rule-only)",
        "precision": 0.612,
        "recall":    0.541,
        "f1":        0.574,
        "auroc":     0.801,
    },
    "Tide_HI": {
        "description": "Tide High-Involvement (rule + simple graph)",
        "precision": 0.718,
        "recall":    0.623,
        "f1":        0.667,
        "auroc":     0.864,
    },
    "MultiGNN": {
        "description": "Multi-GNN (Pareja et al. 2020)",
        "precision": 0.791,
        "recall":    0.712,
        "f1":        0.749,
        "auroc":     0.921,
    },
    "MEGA_GNN": {
        "description": "MEGA-GNN (bidirectional temporal GNN)",
        "precision": 0.843,
        "recall":    0.778,
        "f1":        0.809,
        "auroc":     0.951,
    },
}
