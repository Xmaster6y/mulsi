"""Script to test probing on random data.

Run with:
```
poetry run python -m scripts.random_probing
```
"""

import torch

from mulsi.probe import LinearProbe, SignalCav

#######################################
# HYPERPARAMETERS
#######################################
n_hidden = 100
n_labels = 3
n_train = 200
n_test = 100
seed = 42
#######################################

torch.manual_seed(seed)

X_train = torch.randn(n_train, n_hidden)
Y_train = torch.randint(0, n_labels, (n_train,))
Y_train = torch.nn.functional.one_hot(Y_train, num_classes=n_labels)
X_test = torch.randn(n_test, n_hidden)
Y_test = torch.randint(0, n_labels, (n_test,))
Y_test = torch.nn.functional.one_hot(Y_test, num_classes=n_labels)

cav_probe = SignalCav()
cav_probe.train(X_train, Y_train)
print(f"[INFO] Signal CAV train score: {cav_probe.score(X_train, Y_train)}")
print(f"[INFO] Signal CAV test score: {cav_probe.score(X_test, Y_test)}")

linear_probe = LinearProbe()
linear_probe.train(X_train, Y_train)
print(f"[INFO] Linear probe train score: {linear_probe.score(X_train, Y_train)}")
print(f"[INFO] Linear probe test score: {linear_probe.score(X_test, Y_test)}")
