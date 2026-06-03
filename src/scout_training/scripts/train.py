# Fine-tunes YOLOv8n on SCOUT dataset with MLflow experiment tracking

import os

TRACKING_URI = "sqlite:////home/khit/scout_ws/mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI

from ultralytics import YOLO
import mlflow
import subprocess
import datetime

MODEL_PATH = r"/home/khit/scout_ws/src/scout_training/models/yolov8n.pt"
DATA_CONFIG = r"/home/khit/scout_ws/src/scout_training/data/scout.yaml"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
FREEZE = 10
DEVICE = 0  # RTX 4060

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("scout-transfer-learning")

def on_epoch_end(trainer):
    metrics = trainer.metrics

    # Loss items is a tensor containing
    # 1. Box loss
    # 2. Classification loss
    # 3. DFL loss
    mlflow.log_metrics({
        "train/box_loss": trainer.loss_items[0].item(),
        "train/cls_loss": trainer.loss_items[1].item(),
        "train/dfl_loss": trainer.loss_items[2].item(),
        "val/mAP50": metrics.get("metrics/mAP50(B)", 0), # B for bbox
        "val/mAP50-95": metrics.get("metrics/mAP50-95(B)", 0)
    }, step=trainer.epoch)

def on_train_end(trainer):
    # trainer.best is auto updated
    mlflow.log_artifact(str(trainer.best))

# Load model and attach callbacks
model = YOLO(MODEL_PATH)
model.add_callback("on_fit_epoch_end", on_epoch_end)
model.add_callback("on_train_end", on_train_end)

git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

# Train
with mlflow.start_run(run_name=f"yolov8n_freeze{FREEZE}_lr{LEARNING_RATE}_{timestamp}"):
    mlflow.log_params({
        "model": "yolov8n",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr0": LEARNING_RATE,
        "freeze": FREEZE,
        "imgsz": IMAGE_SIZE,
        "dataset_git_hash": git_hash,
    })
    model.train(
        data=DATA_CONFIG, epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
        lr0=LEARNING_RATE, freeze=FREEZE, device=DEVICE,
        project="/home/khit/scout_ws/src/scout_training/runs", name="finetune_v1",
    )