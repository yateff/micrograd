# first we define the Value class for the scalar values of the net
import numpy as np

class Value():
    def __init__(self, data = 0.0, children = [], op = ''):
        self.data = data
        self.grad = 0.0
        self._prev = set(children)
        self._backward = lambda:None
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += np.log(self.data) * out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self, ), 'tanh')
        def _backward():
            self.grad += (1 - (out.data ** 2)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self, ), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Value(1.0 / (1.0 + np.e ** -self.data), (self, ), 'sigmoid')
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1
    def __truediv__(self, other):
        return self * (other ** -1)
    def __sub__ (self, other):
        return self + (-other)
    def __radd__(self, other):
        return self + Value(other)
    def __rsub__(self, other):
        return Value(other) - self
    def __rmul__(self, other):
        return self * Value(other)
    def __rpow__(self, other):
        return Value(other) ** self
    def __rtruediv__(self, other):
        return Value(other) / self

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
