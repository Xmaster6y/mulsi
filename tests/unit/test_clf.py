"""Test of the clf.
"""

import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mulsi import CLF


@pytest.fixture
def pipe_clf():
    x = torch.rand(10, 4)
    y = torch.rand(10, 3).argmax(dim=1)
    pipe_clf = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression())]
    )
    pipe_clf.fit(x, y)
    return pipe_clf


@pytest.fixture
def torch_clf(pipe_clf):
    return CLF(pipe_clf=pipe_clf, classes=["a", "b", "c"])


class TestCLF:
    def test_forward(self, torch_clf):
        x = torch.rand(10, 4)
        assert torch_clf.forward(x).shape == (10, 3)

    def test_predict(self, pipe_clf, torch_clf):
        x = torch.rand(10, 4)
        sk_pred = torch.tensor(pipe_clf.predict(x))
        torch_pred = torch_clf.predict(x)
        assert (torch_pred == sk_pred).all()

    def test_predict_proba(self, pipe_clf, torch_clf):
        x = torch.rand(10, 4)
        sk_pred = torch.tensor(pipe_clf.predict_proba(x))
        torch_pred = torch_clf.predict_proba(x)
        assert torch.allclose(torch_pred, sk_pred, atol=1e-5)
