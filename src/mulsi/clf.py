"""Classifier class.
"""

from typing import List

import torch


class CLF(torch.nn.Module):
    def __init__(self, pipe_clf, classes: List[str]):
        super().__init__()
        self._classes = classes
        self._scale = torch.tensor(pipe_clf.named_steps["scaler"].scale_)
        self._mean = torch.tensor(pipe_clf.named_steps["scaler"].mean_)
        self._coef = torch.tensor(pipe_clf.named_steps["clf"].coef_)
        self._intercept = torch.tensor(pipe_clf.named_steps["clf"].intercept_)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = (x - self._mean) / self._scale
        return torch.matmul(x, self._coef.T) + self._intercept

    def loss(self, x, labels):
        return torch.nn.functional.cross_entropy(self.forward(x), labels)

    @torch.no_grad()
    def predict(self, x, output_class=False):
        indices = torch.argmax(self.forward(x), dim=1)
        if output_class:
            return indices, [self._classes[i] for i in indices]
        else:
            return indices

    @torch.no_grad()
    def predict_proba(self, x):
        return self.softmax(self.forward(x))
