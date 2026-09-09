"""Evaluate the best ANN checkpoint on the untouched Fashion-MNIST test set."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from fmnist.data import build_loaders
from fmnist.model import FashionMNISTANN
from fmnist.utils import save_json, set_seed

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    artifact_dir = Path(args.artifact_dir)
    checkpoint = torch.load(artifact_dir / "best_model.pt", map_location=device, weights_only=False)

    model = FashionMNISTANN(dropout=float(checkpoint.get("dropout", 0.2))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loaders = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loaders.test:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "test_accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_accuracy": checkpoint.get("validation_accuracy"),
        "class_metrics": report,
    }
    save_json(metrics, artifact_dir / "metrics.json")

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Fashion-MNIST Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(artifact_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"test_accuracy={accuracy:.4f}")
    print(f"macro_precision={precision:.4f} macro_recall={recall:.4f} macro_f1={f1:.4f}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fashion-MNIST ANN")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
