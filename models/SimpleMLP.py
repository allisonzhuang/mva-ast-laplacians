import copy

from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, neurons, activation: nn.Module | None = None, use_bn: bool = False, dropout: float = 0.0, last_activation: bool = False):
        super().__init__()

        if len(neurons) < 2:
            raise ValueError("`neurons` must be a list/tuple like [in_features, ..., out_features] with length >= 2.")

        if activation is None:
            activation = nn.ReLU()

        layers: list[nn.Module] = []
        num_layers = len(neurons) - 1

        for i in range(num_layers):
            in_f, out_f = neurons[i], neurons[i + 1]
            layers.append(nn.Linear(in_f, out_f))

            if last_activation or i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_f))

                layers.append(copy.deepcopy(activation))

                if dropout and dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
