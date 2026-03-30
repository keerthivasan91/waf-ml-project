"""
ml/layer2a/train.py

Trains both Layer 2A candidates on normal-only traffic,
evaluates on a mixed test set, and saves the best model + scaler.

Run this file directly in Colab after notebook 02 has produced
ml/data/splits/  with:
    normal_train.npy, normal_val.npy,
    test_X.npy, test_y.npy   (0=normal, 1=attack)

Usage:
    python train.py
    # or in Colab:
    %run train.py
"""

import numpy as np
import json
from pathlib import Path

from feature_engineering.normalizer import FeatureNormalizer
from candidates.isolation_forest import IsolationForestModel
from candidates.autoencoder_shallow import ShallowAutoencoderModel
from evaluate import evaluate_candidate, pick_best

SPLITS_DIR  = Path("../data/splits")
EXPORT_DIR  = Path("../exported_models")
EXPORT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("Layer 2A — Training both candidates")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    X_train_raw = np.load(SPLITS_DIR / "l2a_normal_train.npy").astype(np.float32)
    X_val_raw   = np.load(SPLITS_DIR / "l2a_normal_val.npy").astype(np.float32)
    X_test_raw  = np.load(SPLITS_DIR / "l2a_test_X.npy").astype(np.float32)
    y_test      = np.load(SPLITS_DIR / "l2a_test_y.npy").astype(int)

    print(f"Train (normal): {X_train_raw.shape}")
    print(f"Val   (normal): {X_val_raw.shape}")
    print(f"Test  (mixed):  {X_test_raw.shape}  |  attack rate: {y_test.mean():.2%}")

    # ── Fit normalizer on training data ──────────────────────────
    norm = FeatureNormalizer()
    X_train = norm.fit_transform(X_train_raw)
    X_val   = norm.transform(X_val_raw)
    X_test  = norm.transform(X_test_raw)
    norm.save(str(EXPORT_DIR / "scaler_l2a.pkl"))

    all_results = []

    # ── Candidate 1: Isolation Forest ────────────────────────────
    print("\n--- Candidate 1: Isolation Forest ---")
    iforest = IsolationForestModel()
    iforest.train(X_train)
    iforest.tune_threshold(X_val, y_val=None)   # unsupervised threshold
    res_if = evaluate_candidate(iforest, X_test, y_test, name="isolation_forest")
    all_results.append(res_if)

    # ── Candidate 2: Shallow Autoencoder ─────────────────────────
    print("\n--- Candidate 2: Shallow Autoencoder ---")
    ae = ShallowAutoencoderModel()
    ae.train(X_train, X_val)
    res_ae = evaluate_candidate(ae, X_test, y_test, name="shallow_autoencoder")
    all_results.append(res_ae)

    # ── Pick winner ───────────────────────────────────────────────
    winner_name, winner_model = pick_best(
        all_results,
        models={"isolation_forest": iforest, "shallow_autoencoder": ae},
        max_fpr=0.05,
    )

    # ── Export winner to ONNX ─────────────────────────────────────
    print(f"\n[train] Exporting winner: {winner_name}")
    winner_model.export_onnx(str(EXPORT_DIR / "layer2a_best.onnx"))

    # ── Save comparison results ───────────────────────────────────
    results_path = EXPORT_DIR / "l2a_results.json"
    with open(results_path, "w") as f:
        json.dump({"results": all_results, "winner": winner_name}, f, indent=2)
    print(f"[train] Results saved → {results_path}")

    print("\n" + "=" * 60)
    print(f"Layer 2A done. Winner: {winner_name.upper()}")
    print("=" * 60)


if __name__ == "__main__":
    main()