# Local WAF retraining

The WAF now supports the same local-machine retraining pattern used by
SpectraSpatialAI.

## Workflow

1. Human-review feedback in `/dashboard/feedback`.
2. Run **Prepare Retraining Batch**.
3. Run **Start Local Retraining**.
4. FastAPI starts a background Python training process on the same machine.
5. The process writes logs and artifacts under
   `ml/retraining_runs/<batch_id>/`.
6. After successful validation, the current model artifacts are backed up.
7. The new L2A/L2B artifacts are promoted and the runtime models are hot-reloaded
   when `LOCAL_RETRAIN_AUTO_PROMOTE=true`.

The selective escalation threshold remains unchanged.

## Required local baseline files

Place these frozen baseline artifacts on the machine:

```
ml/retraining_artifacts/base/
├── layer2b_bigru_checkpoint.pt
├── l2b_train_X_tokens.npy
├── l2b_train_y.npy
├── l2b_val_X_tokens.npy
├── l2b_val_y.npy
├── l2a_normal_val.npy
└── l2a_attack_val.npy
```

The checkpoint should be the trusted BiGRU checkpoint produced by baseline
training. Keep the original train/validation arrays frozen for regression
checks.

## Install

Use the same virtual environment that runs FastAPI:

```bash
pip install -r app/requirements.txt
pip install -r requirements-training.txt
```

For a native local run:

```bash
uvicorn app.main:app --reload --port 8000
```

Then open `/dashboard/retraining`.

## Configuration

Override paths in `.env` when needed:

```env
LOCAL_RETRAIN_ENABLED=true
LOCAL_RETRAIN_AUTO_PROMOTE=true
RETRAIN_BASE_CHECKPOINT=ml/retraining_artifacts/base/layer2b_bigru_checkpoint.pt
RETRAIN_BASE_TRAIN_X=ml/retraining_artifacts/base/l2b_train_X_tokens.npy
RETRAIN_BASE_TRAIN_Y=ml/retraining_artifacts/base/l2b_train_y.npy
RETRAIN_VAL_X=ml/retraining_artifacts/base/l2b_val_X_tokens.npy
RETRAIN_VAL_Y=ml/retraining_artifacts/base/l2b_val_y.npy
RETRAIN_L2A_NORMAL_VAL=ml/retraining_artifacts/base/l2a_normal_val.npy
RETRAIN_L2A_ATTACK_VAL=ml/retraining_artifacts/base/l2a_attack_val.npy
RETRAIN_LOCAL_RUNS_DIR=ml/retraining_runs
```

Absolute paths are supported.

## Run output

Each local training run contains:

```
ml/retraining_runs/<batch_id>/
├── input_batch.json
├── training.log
├── artifacts/
│   ├── layer2b_best.onnx
│   ├── layer2b_bigru_checkpoint.pt
│   ├── layer2a_best.onnx
│   ├── layer2a_best_threshold.txt
│   ├── scaler_l2a.pkl
│   └── retraining_report.json
└── backup_before_promotion/
```

Set `LOCAL_RETRAIN_AUTO_PROMOTE=false` to train and validate locally without
changing the active models.

The selective escalation threshold remains unchanged by this process.
