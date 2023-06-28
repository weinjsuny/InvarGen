from collections import namedtuple

import numpy as np
from invargen.data.expression import *


GenericOperator = namedtuple('GenericOperator', ['name', 'function', 'arity'])

unary_ops = []
binary_ops = [Add, Sub, Mul]

def unary(cls):
    def _calc(a):
        n = len(a)
        return np.array([f'{cls.__name__}({a[i]})' for i in range(n)])

    return _calc


def binary(cls):
    def _calc(a, b):
        n = len(a)
        a = a.astype(str)
        b = b.astype(str)
        return np.array([f'{cls.__name__}({a[i]},{b[i]})' for i in range(n)])

    return _calc



funcs: List[GenericOperator] = []
for op in unary_ops:
    funcs.append(GenericOperator(function=unary(op), name=op.__name__, arity=1))
for op in binary_ops:
    funcs.append(GenericOperator(function=binary(op), name=op.__name__, arity=2))