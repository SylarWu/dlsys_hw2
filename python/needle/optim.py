"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, p in enumerate(self.params):
            if i not in self.u:
                self.u[i] = ndl.init.zeros(*p.shape, dtype=p.dtype)
            if p.grad is None:
                continue
            grad = ndl.Tensor(p.grad.data + self.weight_decay * p.data, dtype=p.dtype)
            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad.data
            p.data = p.data - self.lr * self.u[i]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            if i not in self.m:
                self.m[i] = ndl.init.zeros(*p.shape, dtype=p.dtype).data
            if i not in self.v:
                self.v[i] = ndl.init.zeros(*p.shape, dtype=p.dtype).data

            # 计算梯度（包含weight decay）
            grad = ndl.Tensor(p.grad.data + self.weight_decay * p.data, dtype=p.dtype).data

            # 更新动量
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * grad.data
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (grad.data ** 2).data

            # 偏差修正
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)

            # 更新参数
            p.data = p.data - self.lr * (m_hat.data / (v_hat.data ** 0.5 + self.eps))
        ### END YOUR SOLUTION
