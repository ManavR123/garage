"""GRU Module."""
import copy

import torch
from torch import nn

from garage.experiment import deterministic
from garage.torch import flatten_to_single_vector, NonLinearity


# pytorch v1.6 issue, see https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
# pylint: disable=unused-argument
class GRUModule(nn.Module):
    """Gated Recurrent Unit (GRU) model in pytorch.

    Args:
        input_dim (int): Dimension of the network input.
        hidden_dim (int): Hidden dimension for GRU cell.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_nonlinearity=nn.Tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        layer_normalization=False,
    ):
        super().__init__()
        self._layers = nn.Sequential()

        gru_layer = nn.GRUCell(input_dim, hidden_dim)
        hidden_w_init(gru_layer.weight_ih)
        hidden_w_init(gru_layer.weight_hh)
        hidden_b_init(gru_layer.bias_ih)
        hidden_b_init(gru_layer.bias_hh)
        self._layers.add_module("gru", gru_layer)
        self._layers.add_module("hidden_activation", NonLinearity(hidden_nonlinearity))

        if layer_normalization:
            self._layers.add_module("layer_normalization", nn.LayerNorm(hidden_dim))

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim) shape.

        Returns:
            torch.Tensor: Output values with (N, *, hidden_dim) shape.

        """
        return self._layers(input_val)