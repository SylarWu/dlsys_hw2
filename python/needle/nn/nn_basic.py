"""The module.
"""
from functools import reduce
from typing import List, Callable, Any, Optional, Tuple
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(
            ops.transpose(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype))
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias:
            out = ops.add(out, ops.broadcast_to(self.bias, out.shape))
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        if len(X.shape) <= 2:
            return X
        batch_size = X.shape[0]
        feature_dim = reduce(lambda base, x: base * x, X.shape[1:], 1)
        return ops.reshape(X, shape=(batch_size, feature_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        y_one_hot = init.one_hot(num_classes, y)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        activate_logits = ops.summation(ops.multiply(logits, y_one_hot), axes=(1,))
        return ops.summation(log_sum_exp - activate_logits) / batch_size
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert x.shape[-1] == self.dim
        batch_size, num_features = x.shape

        weight = ops.broadcast_to(ops.reshape(self.weight, (1, num_features)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, num_features)), x.shape)

        if not self.training:
            running_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, num_features)), x.shape)
            running_var = ops.broadcast_to(ops.reshape(self.running_var, (1, num_features)), x.shape)
        else:
            mean, variance = _calc_mean_and_variance(x, (0,), keep_dims=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance

            running_mean = ops.broadcast_to(ops.reshape(mean, (1, num_features)), x.shape)
            running_var = ops.broadcast_to(ops.reshape(variance, (1, num_features)), x.shape)
        out = (x - running_mean) / ((running_var + self.eps) ** 0.5)
        return weight * out + bias
        ### END YOUR SOLUTION

def _calc_mean_and_variance(x: Tensor, axes: Optional[Tuple[int]] = None, keep_dims=True) -> Tensor:
    def _broadcast_to_x_shape(input: Tensor, match_shape: Tuple[int, ...], x_shape: Tuple[int, ...]) -> Tensor:
        return ops.broadcast_to(ops.reshape(input, match_shape), x_shape)
    axis = axes if axes is not None else tuple(range(len(x.shape)))
    scalar = 1
    for i in axis:
        scalar *= x.shape[i]
    match_shape = tuple([1 if i in axis else n for i, n in enumerate(x.shape)])

    mean = ops.summation(x, axes=axis) / scalar
    variance = ops.summation((x - _broadcast_to_x_shape(mean, match_shape, x.shape)) ** 2, axes=axis) / scalar
    if keep_dims:
        return ops.reshape(mean, match_shape), ops.reshape(variance, match_shape)
    else:
        return mean, variance

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert self.dim == x.shape[-1], f"Error dimension: input dim is {x.shape[-1]} but got {self.dim}"
        axis = (len(x.shape) - 1,)

        weight = ops.broadcast_to(ops.reshape(self.weight, tuple(1 for _ in range(len(x.shape) - 1)) + (self.dim,)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, tuple(1 for _ in range(len(x.shape) - 1)) + (self.dim,)), x.shape)

        mean, variance = _calc_mean_and_variance(x, axis, keep_dims=True)
        mean, variance = ops.broadcast_to(mean, x.shape), ops.broadcast_to(variance, x.shape)

        return weight * ((x - mean) / ((variance + self.eps) ** 0.5)) + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_mask = init.randb(*x.shape, p=1 - self.p)
            x = ops.divide_scalar(ops.multiply(x, drop_mask), 1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
