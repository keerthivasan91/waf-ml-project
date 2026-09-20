import asyncio
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.db.mongodb import connect_db, close_db, get_db
from app.core.config import settings


BATCH_ID = "7614b7fc-c913-4574-96a8-f53bdf60c6d7"
SEED = 42
MAX_RATIO = settings.RETRAIN_MAX_BATCH_RATIO

LABELS = ["normal", "sqli", "xss", "lfi", "other_attack"]


async def main():
    await connect_db()

    try:
        db = get_db()

        batch = await db.retrain_batches.find_one(
            {"batch_id": BATCH_ID},
            {"_id": 0},
        )

        if not batch:
            raise RuntimeError(f"Batch not found: {BATCH_ID}")

        samples = batch.get("samples", [])
        print(f"Original clean samples: {len(samples)}")

        base_y = np.load(
            settings.RETRAIN_BASE_TRAIN_Y
        ).astype(np.int64)

        base_counts = dict(
            zip(
                *np.unique(
                    base_y,
                    return_counts=True,
                )
            )
        )

        label_to_id = {
            "normal": 0,
            "sqli": 1,
            "xss": 2,
            "lfi": 3,
            "other_attack": 4,
            "false_positive": 0,
        }

        grouped = {label: [] for label in LABELS}

        for sample in samples:
            label = sample.get("verified_label")

            if label == "false_positive":
                label = "normal"

            if label not in grouped:
                print("Skipping unknown label:", label)
                continue

            grouped[label].append(sample)

        rng = random.Random(SEED)
        selected = []

        print("\nClass gate:")

        for label in LABELS:
            class_id = label_to_id[label]
            base_count = int(base_counts.get(class_id, 0))
            max_allowed = int(base_count * MAX_RATIO)

            current = grouped[label]
            rng.shuffle(current)

            keep = min(len(current), max_allowed)
            chosen = current[:keep]

            selected.extend(chosen)

            ratio = keep / base_count if base_count else 0.0

            print(
                f"  {label:<14} "
                f"base={base_count:<5} "
                f"current={len(current):<5} "
                f"keep={keep:<5} "
                f"ratio={ratio:.4f}"
            )

        if len(selected) < settings.RETRAIN_MIN_SAMPLES:
            raise RuntimeError(
                f"Capped batch has only {len(selected)} samples; "
                f"minimum is {settings.RETRAIN_MIN_SAMPLES}"
            )

        new_batch_id = str(uuid4())
        now = datetime.utcnow()

        new_batch = {
            "batch_id": new_batch_id,
            "created_at": now,
            "status": "queued",
            "n_raw": batch.get("n_raw", len(samples)),
            "n_clean": len(selected),
            "n_rejected": batch.get("n_rejected", 0),
            "reject_reason_breakdown": batch.get(
                "reject_reason_breakdown",
                {},
            ),
            "samples": selected,
            "derived_from_batch": BATCH_ID,
            "note": (
                "Demo-derived batch capped per class to the configured "
                "10% retraining-size gate."
            ),
        }

        await db.retrain_batches.insert_one(new_batch)

        await db.retrain_batches.update_one(
            {"batch_id": BATCH_ID},
            {
                "$set": {
                    "status": "superseded",
                    "superseded_by": new_batch_id,
                }
            },
        )

        print("\nCreated new batch:")
        print(f"  batch_id = {new_batch_id}")
        print(f"  n_clean  = {len(selected)}")

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())