import torch

from fmnist.model import FashionMNISTANN
from fmnist.utils import EarlyStopping


def test_model_output_shape():
    model = FashionMNISTANN(dropout=0.2)
    model.eval()
    inputs = torch.randn(4, 1, 28, 28)
    outputs = model(inputs)
    assert outputs.shape == (4, 10)


def test_early_stopping_after_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    assert stopper.step(1.0) is False
    assert stopper.step(1.1) is False
    assert stopper.step(1.2) is True


def test_single_training_epoch_runs():
    from torch import nn
    from torch.optim import SGD
    from torch.utils.data import DataLoader, TensorDataset

    from fmnist.train import run_epoch

    model = FashionMNISTANN(dropout=0.0)
    loader = DataLoader(
        TensorDataset(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))),
        batch_size=8,
    )
    optimizer = SGD(model.parameters(), lr=1e-3)
    loss, accuracy = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
