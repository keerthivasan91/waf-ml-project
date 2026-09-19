# Offline adaptive retraining

This is the offline stage of the WAF feedback loop.

## 1. Prepare a batch

Open /dashboard/retraining.

Run "Prepare Retraining Batch". The backend validates the reviewed samples
against its anti-poisoning checks and stores the clean batch.

Download the resulting batch from:

/api/feedback/retrain-batches/{batch_id}/export

## 2. Prepare Kaggle or Colab

You need:

- the exported retraining JSON
- a trusted Layer 2B BiGRU checkpoint
- the frozen original L2B train and validation token arrays
- the frozen original L2A normal and attack validation arrays
- the currently deployed model artifacts

The offline script deliberately consumes the frozen validation data. It does
not silently re-split the original benchmark.

## 3. Install training dependencies

    pip install numpy scipy scikit-learn pandas torch onnx onnxruntime onnxscript joblib

## 4. Run offline training

Example:

    python -m ml.retraining.offline_retrain \
      --batch waf_retrain_BATCH_ID.json \
      --base-checkpoint /kaggle/input/waf-checkpoint/layer2b_bigru_checkpoint.pt \
      --base-model-dir ml/exported_models \
      --base-train-x /kaggle/input/hiwaf-split-v1/data/splits/l2b_train_X_tokens_sqli8k.npy \
      --base-train-y /kaggle/input/hiwaf-split-v1/data/splits/l2b_train_y_sqli8k.npy \
      --val-x /kaggle/input/hiwaf-split-v1/data/splits/l2b_val_X_tokens.npy \
      --val-y /kaggle/input/hiwaf-split-v1/data/splits/l2b_val_y.npy \
      --l2a-normal-val /kaggle/input/hiwaf-split-v1/data/splits/l2a_normal_val.npy \
      --l2a-attack-val /kaggle/input/hiwaf-split-v1/data/splits/l2a_attack_val.npy \
      --max-batch-ratio 0.10 \
      --output-dir ml/exported_models/retrained_BATCH_ID

The script:

1. splits the clean batch by URL family into fine-tune and holdout data;
2. fine-tunes Layer 2B from the trusted checkpoint;
3. selects the checkpoint using the frozen validation set;
4. refuses deployment when validation macro-F1 falls beyond the configured tolerance;
5. enforces the 10% per-class feedback/training-size gate used by the adaptive
   retraining design;
6. evaluates held-out feedback samples after model selection;
6. recalibrates Layer 2A's own operating threshold;
7. keeps the selective escalation threshold unchanged;
8. exports a new Layer 2B ONNX model, checkpoint, threshold and report.

## 5. Promote only after review

Do not overwrite production artifacts immediately.

Inspect retraining_report.json first. Then copy the validated artifacts to the
deployment ml/exported_models directory:

- layer2b_best.onnx
- layer2b_bigru_checkpoint.pt
- layer2a_best.onnx
- layer2a_best.onnx.data (when present)
- layer2a_best_threshold.txt
- scaler_l2a.pkl

Then use "Reload Models" on /dashboard/retraining.

This offline job never changes the selective escalation threshold.
