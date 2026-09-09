"""Train the production ANN on Fashion-MNIST."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fmnist.data import build_loaders
from fmnist.model import FashionMNISTANN
from fmnist.utils import EarlyStopping, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = FashionMNISTANN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "best_model.pt"
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    started = time.time()

    print(f"device={device} | train={len(loaders.train.dataset)} | val={len(loaders.validation.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders.train, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders.validation, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate": current_lr,
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "FashionMNISTANN",
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

        if stopper.step(val_loss):
            print(f"early_stopping=true at epoch={epoch}")
            break

    summary = {
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
        "training_seconds": time.time() - started,
        "config": vars(args),
        "history": history,
    }
    save_json(summary, artifact_dir / "training_history.json")

    epochs_axis = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_axis, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs_axis, [row["validation_loss"] for row in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "loss_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_axis, [row["train_accuracy"] for row in history], label="train")
    ax.plot(epochs_axis, [row["validation_accuracy"] for row in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="Training and Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_curve.png", dpi=160)
    plt.close(fig)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST ANN")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
