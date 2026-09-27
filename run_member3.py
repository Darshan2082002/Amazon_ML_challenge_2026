import json
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation.evaluate import compute_macro_f05

OUTPUT_DIR = BASE_DIR / "output"
PRED_PATH = OUTPUT_DIR / "prediction_details.tsv"


def main():
    print("============================================")
    print("      MEMBER 3 - EVALUATION PIPELINE        ")
    print("============================================")

    if not PRED_PATH.exists():
        print(f"[ERROR] Prediction details file not found: {PRED_PATH}")
        sys.exit(1)

    print(f"Loading prediction details from: {PRED_PATH}")
    pred_df = pd.read_csv(PRED_PATH, sep="\t")

    metrics = compute_macro_f05(pred_df, beta=0.5)

    print("\n--------------------------------------------")
    print("              EVALUATION METRICS            ")
    print("--------------------------------------------")
    print(f" Micro Precision : {metrics['micro_precision']:.4f}")
    print(f" Micro Recall    : {metrics['micro_recall']:.4f}")
    print(f" Micro F0.5      : {metrics['micro_f0.5']:.4f}")
    print("--------------------------------------------")
    print(f" Macro Precision : {metrics['macro_precision']:.4f}")
    print(f" Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f" Macro F0.5      : {metrics['macro_f0.5']:.4f}")
    print("--------------------------------------------\n")

    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics summary to: {metrics_path}")


if __name__ == "__main__":
    main()