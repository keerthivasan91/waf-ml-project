"""Production offline retraining pipeline for WAF-ML.

Consumes a clean, human-verified batch exported from
/api/feedback/retrain-batches/{batch_id}/export and produces versioned
model artifacts for manual promotion.

Procedure:
  1. Load the verified batch.
  2. Split it by URL family into fine-tune/holdout sets.
  3. Fine-tune the accepted Layer 2B BiGRU from a trusted checkpoint.
  4. Select the model using the original validation set, never the holdout.
  5. Evaluate the held-out feedback samples for a generalization check.
  6. Recalibrate Layer 2A's operating threshold using validation data.
  7. Export a new Layer 2B ONNX model, checkpoint, threshold and report.

The selective escalation threshold is deliberately unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from ml.feature_engineering.extractor import extract_features, to_vector, strip_to_path_query
from ml.feature_engineering.normalizer import Normalizer
from ml.feature_engineering.tokenizer import CharTokenizer
from ml.layer2b.candidates.gru import GRUClassifier


CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
LABEL_TO_ID["false_positive"] = 0


def canonical_family(url: str) -> str:
    text = re.sub(r"%[0-9a-fA-F]{2}", "", (url or "").lower())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_batch(path: Path, min_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Retraining batch JSON must contain a 'samples' list.")
    if len(samples) < min_samples:
        raise ValueError(
            f"Only {len(samples)} clean samples are present; minimum is {min_samples}."
        )

    valid = []
    for row in samples:
        label = row.get("verified_label")
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unsupported verified_label in batch: {label!r}")
        valid.append(row)
    return payload, valid


def split_batch(samples: list[dict[str, Any]], holdout_fraction: float, seed: int):
    groups = np.array([canonical_family(s.get("url", "")) for s in samples])
    idx = np.arange(len(samples))

    if len(np.unique(groups)) >= 4:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=holdout_fraction,
            random_state=seed,
        )
        fit_idx, holdout_idx = next(splitter.split(idx, groups=groups))
    else:
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(idx)
        cut = max(1, int(round(len(samples) * (1.0 - holdout_fraction))))
        fit_idx, holdout_idx = shuffled[:cut], shuffled[cut:]

    if len(holdout_idx) == 0:
        raise ValueError("Holdout split is empty; increase batch size.")
    return [samples[i] for i in fit_idx], [samples[i] for i in holdout_idx]


def request_to_tokens(samples: list[dict[str, Any]]) -> np.ndarray:
    tokenizer = CharTokenizer(max_len=512)
    texts = [
        f"{s.get('method', 'GET')} "
        f"{strip_to_path_query(str(s.get('url', '')))} "
        f"{s.get('body', '')}"
        for s in samples
    ]
    return tokenizer.encode_batch(texts).astype(np.int64)


def request_to_labels(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([LABEL_TO_ID[s["verified_label"]] for s in samples], dtype=np.int64)


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model_from_checkpoint(checkpoint_path: Path) -> GRUClassifier:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    params = ckpt.get("train_params") or ckpt.get("params") or {}
    model = GRUClassifier(
        vocab_size=int(ckpt.get("vocab_size", 100)),
        embed_dim=int(params.get("embed_dim", 64)),
        hidden_dim=int(params.get("hidden_dim", 128)),
        num_layers=int(params.get("num_layers", 2)),
        num_classes=int(params.get("num_classes", len(CLASS_NAMES))),
        dropout=float(params.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def predict_tokens(model: GRUClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, len(X), 256):
        xb = torch.from_numpy(X[start:start + 256]).long().to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_macro_f1(
    model: GRUClassifier, X: np.ndarray, y: np.ndarray, device: torch.device
) -> float:
    return float(
        f1_score(
            y,
            predict_tokens(model, X, device),
            average="macro",
            zero_division=0,
        )
    )


def validate_batch_ratio(
    samples: list[dict[str, Any]],
    base_train_y: np.ndarray,
    max_ratio: float,
) -> dict[str, float]:
    base_counts = np.bincount(base_train_y, minlength=len(CLASS_NAMES)).astype(int)
    batch_counts = np.zeros(len(CLASS_NAMES), dtype=int)
    for sample in samples:
        batch_counts[LABEL_TO_ID[sample["verified_label"]]] += 1

    ratios: dict[str, float] = {}
    violations = []
    for idx, name in enumerate(CLASS_NAMES):
        denom = max(1, int(base_counts[idx]))
        ratio = float(batch_counts[idx] / denom)
        ratios[name] = ratio
        if ratio > max_ratio:
            violations.append(
                f"{name}: {batch_counts[idx]}/{denom}={ratio:.4f} > {max_ratio:.4f}"
            )

    if violations:
        raise RuntimeError(
            "Retraining batch exceeds the per-class size gate: " + "; ".join(violations)
        )

    return ratios


def fine_tune_l2b(
    base_checkpoint: Path,
    base_train_x: Path,
    base_train_y: Path,
    val_x: Path,
    val_y: Path,
    fit_samples: list[dict[str, Any]],
    holdout_samples: list[dict[str, Any]],
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    oversample_factor: int,
    tolerance: float,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_base = np.load(base_train_x).astype(np.int64)
    y_base = np.load(base_train_y).astype(np.int64)
    X_val = np.load(val_x).astype(np.int64)
    y_val = np.load(val_y).astype(np.int64)

    X_feedback = request_to_tokens(fit_samples)
    y_feedback = request_to_labels(fit_samples)

    X_feedback = np.repeat(X_feedback, oversample_factor, axis=0)
    y_feedback = np.repeat(y_feedback, oversample_factor, axis=0)

    X_train = np.vstack([X_base, X_feedback])
    y_train = np.concatenate([y_base, y_feedback])

    X_holdout = request_to_tokens(holdout_samples)
    y_holdout = request_to_labels(holdout_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(base_checkpoint).to(device)

    baseline_f1 = evaluate_macro_f1(model, X_val, y_val, device)
    baseline_holdout_pred = predict_tokens(model, X_holdout, device)
    baseline_holdout_accuracy = float(np.mean(baseline_holdout_pred == y_holdout))
    baseline_holdout_f1 = float(
        f1_score(y_holdout, baseline_holdout_pred, average="macro", zero_division=0)
    )

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).long(),
            torch.from_numpy(y_train).long(),
        ),
        batch_size=128,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_state = None
    best_val_f1 = -1.0
    epoch_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        val_f1 = evaluate_macro_f1(model, X_val, y_val, device)
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(loader)),
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 >= baseline_f1 - tolerance and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(
            "No fine-tuned epoch stayed within the validation F1 tolerance. "
            "Do not deploy this batch; inspect the retraining data/model."
        )

    model.load_state_dict(best_state)
    model.eval()

    holdout_pred = predict_tokens(model, X_holdout, device)
    holdout_accuracy = float(np.mean(holdout_pred == y_holdout))
    holdout_f1 = float(
        f1_score(y_holdout, holdout_pred, average="macro", zero_division=0)
    )

    checkpoint_path = output_dir / "layer2b_bigru_checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": {
                "embed_dim": 64,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "num_classes": len(CLASS_NAMES),
            },
            "vocab_size": int(model.embedding.num_embeddings),
            "source": str(base_checkpoint),
        },
        checkpoint_path,
    )

    model_cpu = model.to("cpu")
    model_cpu.eval()
    dummy = torch.zeros(1, 512, dtype=torch.long)
    onnx_path = output_dir / "layer2b_best.onnx"
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["token_ids"],
        output_names=["logits"],
        dynamic_axes={"token_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    session = ort.InferenceSession(str(onnx_path))
    session.run(None, {"token_ids": dummy.numpy()})

    return {
        "baseline_val_macro_f1": baseline_f1,
        "selected_val_macro_f1": best_val_f1,
        "baseline_holdout_accuracy": baseline_holdout_accuracy,
        "baseline_holdout_macro_f1": baseline_holdout_f1,
        "holdout_accuracy": holdout_accuracy,
        "holdout_macro_f1": holdout_f1,
        "fit_samples": len(fit_samples),
        "holdout_samples": len(holdout_samples),
        "oversample_factor": oversample_factor,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "epoch_history": epoch_history,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
    }


def calibrate_l2a(
    l2a_onnx: Path,
    scaler_path: Path,
    normal_val_path: Path,
    attack_val_path: Path,
    samples: list[dict[str, Any]],
    output_path: Path,
    fpr_cap: float,
) -> dict[str, Any]:
    session = ort.InferenceSession(str(l2a_onnx))
    input_name = session.get_inputs()[0].name
    scaler = Normalizer.load(str(scaler_path))

    X_normal = scaler.transform(np.load(normal_val_path).astype(np.float32))
    X_attack = scaler.transform(np.load(attack_val_path).astype(np.float32))

    verified_attack = [
        s for s in samples
        if s.get("verified_label") not in {"normal", "false_positive"}
    ]
    if verified_attack:
        extra = np.vstack(
            [
                scaler.transform(
                    to_vector(
                        extract_features(
                            {
                                "url": str(s.get("url", "")),
                                "method": str(s.get("method", "GET")),
                                "headers": {},
                                "body": str(s.get("body", "")),
                            }
                        )
                    )
                )
                for s in verified_attack
            ]
        )
        X_attack = np.vstack([X_attack, extra])

    def scores(X: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(X), 256):
            xb = X[start:start + 256].astype(np.float32)
            recon = session.run(None, {input_name: xb})[0]
            out.append(np.mean((xb - recon) ** 2, axis=1))
        return np.concatenate(out)

    normal_scores = scores(X_normal)
    attack_scores = scores(X_attack)

    lo, hi = np.percentile(normal_scores, [50, 99])
    thresholds = np.linspace(lo, hi, 500)
    candidates = [
        (
            threshold,
            float(np.mean(normal_scores > threshold)),
            float(np.mean(attack_scores > threshold)),
        )
        for threshold in thresholds
    ]
    valid = [row for row in candidates if row[1] <= fpr_cap]
    if not valid:
        raise RuntimeError("No L2A threshold satisfies the requested FPR cap.")

    threshold, fpr, recall = max(
        valid,
        key=lambda row: (row[2], -row[1], -row[0]),
    )

    threshold_before = None
    if output_path.exists():
        try:
            threshold_before = float(output_path.read_text(encoding="utf-8").strip())
        except ValueError:
            threshold_before = None

    output_path.write_text(f"{threshold:.8f}\n", encoding="utf-8")

    return {
        "threshold_before": threshold_before,
        "threshold_after": threshold,
        "validation_fpr": fpr,
        "validation_recall": recall,
        "attack_validation_size": int(len(X_attack)),
    }


def copy_base_artifacts(base_model_dir: Path, output_dir: Path) -> None:
    for name in (
        "layer2a_best.onnx",
        "layer2a_best.onnx.data",
        "scaler_l2a.pkl",
    ):
        src = base_model_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline WAF-ML adaptive retraining")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, default=Path("ml/exported_models"))
    parser.add_argument("--base-train-x", type=Path, required=True)
    parser.add_argument("--base-train-y", type=Path, required=True)
    parser.add_argument("--val-x", type=Path, required=True)
    parser.add_argument("--val-y", type=Path, required=True)
    parser.add_argument("--l2a-normal-val", type=Path, required=True)
    parser.add_argument("--l2a-attack-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--oversample-factor", type=int, default=5)
    parser.add_argument("--val-f1-tolerance", type=float, default=0.005)
    parser.add_argument("--l2a-fpr-cap", type=float, default=0.05)
    parser.add_argument("--max-batch-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.holdout_fraction < 0.5:
        raise ValueError("--holdout-fraction must be between 0 and 0.5.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_meta, samples = load_batch(args.batch, args.min_samples)
    base_y_for_gate = np.load(args.base_train_y).astype(np.int64)
    batch_ratios = validate_batch_ratio(samples, base_y_for_gate, args.max_batch_ratio)
    fit_samples, holdout_samples = split_batch(samples, args.holdout_fraction, args.seed)

    copy_base_artifacts(args.base_model_dir, args.output_dir)

    l2b_report = fine_tune_l2b(
        base_checkpoint=args.base_checkpoint,
        base_train_x=args.base_train_x,
        base_train_y=args.base_train_y,
        val_x=args.val_x,
        val_y=args.val_y,
        fit_samples=fit_samples,
        holdout_samples=holdout_samples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        oversample_factor=args.oversample_factor,
        tolerance=args.val_f1_tolerance,
        seed=args.seed,
    )

    l2a_report = calibrate_l2a(
        l2a_onnx=args.base_model_dir / "layer2a_best.onnx",
        scaler_path=args.base_model_dir / "scaler_l2a.pkl",
        normal_val_path=args.l2a_normal_val,
        attack_val_path=args.l2a_attack_val,
        samples=samples,
        output_path=args.output_dir / "layer2a_best_threshold.txt",
        fpr_cap=args.l2a_fpr_cap,
    )

    report = {
        "batch_id": batch_meta.get("batch_id"),
        "batch_created_at": batch_meta.get("created_at"),
        "input_samples": len(samples),
        "fit_samples": len(fit_samples),
        "holdout_samples": len(holdout_samples),
        "batch_class_ratios": batch_ratios,
        "l2a": l2a_report,
        "l2b": l2b_report,
        "selective_escalation_threshold_changed": False,
        "artifacts_dir": str(args.output_dir.resolve()),
    }

    (args.output_dir / "retraining_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))
    print("\nOffline retraining completed. Review retraining_report.json before promotion.")


if __name__ == "__main__":
    main()
