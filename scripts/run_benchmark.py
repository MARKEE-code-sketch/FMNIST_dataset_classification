"""One-command training + test evaluation for the production ANN."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from fmnist.evaluate import evaluate
from fmnist.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Fashion-MNIST ANN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train_args = SimpleNamespace(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        validation_fraction=0.1,
        patience=args.patience,
        min_delta=1e-4,
        seed=args.seed,
        num_workers=0,
        cpu=args.cpu,
    )
    train(train_args)

    eval_args = SimpleNamespace(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        batch_size=256,
        validation_fraction=0.1,
        seed=args.seed,
        num_workers=0,
        cpu=args.cpu,
    )
    evaluate(eval_args)


if __name__ == "__main__":
    main()
