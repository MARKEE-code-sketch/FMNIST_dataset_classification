# Fashion-MNIST Classification with PyTorch

A reproducible deep-learning project for **10-class Fashion-MNIST image classification**. The repository preserves the original Colab experiments and adds an installable, testable ANN training/evaluation pipeline for local CPU/GPU environments.

## Why this repository exists

The original work was notebook-first and depended on a Fashion-MNIST CSV stored in a personal Google Drive path. This version keeps those notebooks as experiment history, but moves the final ANN workflow into reusable Python modules with a deterministic data split, checkpointing, early stopping, LR scheduling, and test-set evaluation.

## Highlights

- **PyTorch ANN:** `784 → 128 → 64 → 10`
- Batch Normalization + ReLU + Dropout regularization
- `CrossEntropyLoss` with label smoothing and raw logits
- AdamW optimizer with weight decay
- `ReduceLROnPlateau` learning-rate scheduling
- Early stopping based on validation loss
- Deterministic seed control
- Official Fashion-MNIST test set kept separate from training/validation
- Accuracy, macro precision, macro recall, macro F1 and class-wise metrics
- Confusion matrix + training/validation curves saved as artifacts
- Pytest coverage for model shape and early-stopping behavior
- GitHub Actions workflow for automated tests

## Repository structure

```text
FMNIST_dataset_classification/
├── .github/
│   └── workflows/
│       └── tests.yml
├── notebooks/
│   ├── advanced_FMNIST_dataset_ANN.ipynb
│   ├── FMNIST_dataset_CNN.ipynb
│   └── FMNIST_dataset_transfer_learning.ipynb
├── scripts/
│   └── run_benchmark.py
├── notebooks/
│   └── production_ann_benchmark.ipynb  # GPU benchmark runner
├── src/
│   └── fmnist/
│       ├── __init__.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── tests/
│   └── test_model.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## ANN architecture

```text
28 × 28 grayscale image
        ↓
      Flatten
        ↓
Linear(784, 128)
BatchNorm1d(128)
ReLU
Dropout(0.2)
        ↓
Linear(128, 64)
BatchNorm1d(64)
ReLU
Dropout(0.2)
        ↓
Linear(64, 10)
        ↓
    Class logits
```

Softmax is deliberately omitted from the model because `CrossEntropyLoss` expects raw logits.

## Data protocol

`torchvision.datasets.FashionMNIST` downloads the canonical dataset directly.

- **60,000 training images** are split deterministically into 90% train / 10% validation.
- **10,000 official test images** remain untouched until final evaluation.
- Inputs are converted to tensors and normalized with Fashion-MNIST dataset statistics.

This replaces the original machine-specific Google Drive CSV path and makes the pipeline reproducible after cloning the repository.

## Installation

```bash
git clone https://github.com/MARKEE-code-sketch/FMNIST_dataset_classification.git
cd FMNIST_dataset_classification

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the complete benchmark

```bash
python scripts/run_benchmark.py
```

The command trains the ANN, reloads the **best validation-loss checkpoint**, and evaluates it once on the official test set.

Useful overrides:

```bash
python scripts/run_benchmark.py --epochs 30 --batch-size 128 --learning-rate 0.001
```


## Run on Google Colab

Open `notebooks/production_ann_benchmark.ipynb` in Colab after these production files are merged into the repository. The notebook clones the repo, runs the tests, executes the complete benchmark on a GPU runtime, prints the final metrics, and renders the training curves/confusion matrix.

## Train and evaluate separately

Train:

```bash
python -m fmnist.train
```

Evaluate the best checkpoint:

```bash
python -m fmnist.evaluate
```

## Generated artifacts

A completed benchmark creates:

```text
artifacts/
├── best_model.pt
├── training_history.json
├── metrics.json
├── loss_curve.png
├── accuracy_curve.png
└── confusion_matrix.png
```

`metrics.json` is the **source of truth** for any metric used in a resume, research report, or project description.

## Results policy

The original ANN notebook contains historical Optuna runs whose logged validation accuracy reaches approximately **88.78%**. It also contains stale notebook output where a later `study.best_params` cell points to an earlier trial. Because of that inconsistency, notebook outputs are treated only as experiment history and **not as the final test benchmark**.

The final headline metric should only be added here after a clean run of `scripts/run_benchmark.py`, using the generated `artifacts/metrics.json`.

## Original experimentation

The notebooks remain valuable because they document the progression of the work:

- `advanced_FMNIST_dataset_ANN.ipynb` — ANN design, regularization, optimizer experiments and Optuna search
- `FMNIST_dataset_CNN.ipynb` — CNN-based Fashion-MNIST experiment
- `FMNIST_dataset_transfer_learning.ipynb` — transfer-learning experiment

The production package does not delete or rewrite them.

## Testing

```bash
pytest
```

CI runs the same test suite automatically on pushes and pull requests.

## Fashion-MNIST classes

`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.

## Next experiment

Once the ANN benchmark is locked, the next useful comparison is to run the existing CNN under the **same train/validation/test protocol** and compare accuracy, macro F1, parameter count, and inference cost rather than comparing unrelated notebook runs.
