"""
ml/layer2b/export_onnx.py

Standalone ONNX export + latency validation for Layer 2B.
Called by train.py automatically; run manually to re-export.

Usage:
    python export_onnx.py --model gru \
                          --weights ../exported_models/gru_weights.pt \
                          --out     ../exported_models/layer2b_best.onnx
"""

import argparse
import numpy as np
import onnxruntime as ort
import time

CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack"]


def validate_onnx(path: str, input_name: str, dummy_input: np.ndarray,
                  target_ms: float = 20.0, n_runs: int = 200) -> dict:
    """Run latency benchmark and confirm output shape."""
    sess  = ort.InferenceSession(path)
    times = []

    # warmup
    for _ in range(10):
        sess.run(None, {input_name: dummy_input})

    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - t0) * 1000)

    times  = sorted(times)
    mean   = round(np.mean(times), 3)
    p95    = round(np.percentile(times, 95), 3)
    p99    = round(np.percentile(times, 99), 3)
    status = "PASS" if p99 <= target_ms else f"WARN (p99={p99}ms > {target_ms}ms target)"

    print(f"[export_onnx] Output shape: {out[0].shape}  num_classes: {out[0].shape[-1]}")
    print(f"[export_onnx] Latency — mean={mean}ms  p95={p95}ms  p99={p99}ms")
    print(f"[export_onnx] Target <{target_ms}ms: {status}")

    return {"mean_ms": mean, "p95_ms": p95, "p99_ms": p99, "pass": p99 <= target_ms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["xgboost", "cnn_1d", "gru"], default="gru")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out",     type=str, default="../exported_models/layer2b_best.onnx")
    args = parser.parse_args()

    if args.model == "xgboost":
        from candidates.xgboost_model import XGBoostModel
        m = XGBoostModel()
        m.load(args.weights)
        m.export_onnx(args.out)
        dummy = np.random.randn(1, 20).astype(np.float32)
        validate_onnx(args.out, "features", dummy)

    elif args.model == "cnn_1d":
        import torch
        from candidates.cnn_1d import CNN1D, MAX_LEN
        net = CNN1D()
        ck  = torch.load(args.weights, map_location="cpu")
        net.load_state_dict(ck["state_dict"])
        net.eval()
        dummy_t = torch.zeros(1, MAX_LEN, dtype=torch.long)
        torch.onnx.export(net, dummy_t, args.out,
                          input_names=["token_ids"], output_names=["logits"],
                          dynamic_axes={"token_ids": {0: "batch"},
                                        "logits":    {0: "batch"}},
                          opset_version=17)
        print(f"[export_onnx] CNN1D exported → {args.out}")
        validate_onnx(args.out, "token_ids", dummy_t.numpy())

    else:  # gru
        import torch
        from candidates.gru import GRUClassifier, MAX_LEN
        net = GRUClassifier()
        ck  = torch.load(args.weights, map_location="cpu")
        net.load_state_dict(ck["state_dict"])
        net.eval()
        dummy_t = torch.zeros(1, MAX_LEN, dtype=torch.long)
        torch.onnx.export(net, dummy_t, args.out,
                          input_names=["token_ids"], output_names=["logits"],
                          dynamic_axes={"token_ids": {0: "batch"},
                                        "logits":    {0: "batch"}},
                          opset_version=17)
        print(f"[export_onnx] GRU exported → {args.out}")
        validate_onnx(args.out, "token_ids", dummy_t.numpy())