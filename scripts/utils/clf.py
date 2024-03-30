"""Classifier class.
"""

from typing import List

import torch


class CLF(torch.nn.Module):
    def __init__(self, n_hidden: int, classes: List[str]):
        super().__init__()
        self._n_classes = len(classes)
        self._classes = classes
        self._n_hidden = n_hidden
        self.linear = torch.nn.Linear(n_hidden, self._n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.linear(x)

    def loss(self, x, labels):
        return torch.nn.functional.cross_entropy(self.forward(x), labels)

    @torch.no_grad()
    def predict(self, x):
        indices = torch.argmax(self.forward(x), dim=1)
        return [self._classes[i] for i in indices]

    @torch.no_grad()
    def predict_proba(self, x):
        return self.softmax(self.forward(x))
