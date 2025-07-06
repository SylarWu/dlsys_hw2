import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    return randn(
        fan_in, fan_out,
        mean=0.0,
        std=gain * math.sqrt(2.0 / (fan_in + fan_out)),
        **kwargs
    )
    ### END YOUR SOLUTION


def gain_factor(nonlinearity="relu"):
    default = math.sqrt(2.0)
    mapping = {
        "relu": default,
    }
    return default if nonlinearity not in mapping else mapping[nonlinearity]


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = gain_factor(nonlinearity) * math.sqrt(3.0 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    return randn(
        fan_in, fan_out,
        mean=0.0,
        std=gain_factor(nonlinearity) / math.sqrt(fan_in),
        **kwargs
    )
    ### END YOUR SOLUTION
