import numpy as np

from pipelines import *


class Circular_Array:
    N = -1
    arr = np.array([])
    curr = 0
    added = 0

    def __init__(self, sz):
        self.N = sz
        self.arr = np.zeros(sz)

    def add(self, element):
        self.arr[self.curr] = element
        self.curr = (self.curr + 1) % self.N
        self.added = self.added + 1

    def get_average(self):
        if self.added < self.N:
            return sum(self.arr[:self.added]) / float(self.arr[:self.added].size)
        else:
            return sum(self.arr) / float(self.arr.size)


def swap(a, b):
    tmp = a
    a = b
    b = tmp

    return a, b

def to_idx(clazz):
    if clazz == type(EmptyPipeline):
        return 0
    # elif clazz == type(ColorPipeline):
    #     return 1
    elif clazz == type(NeuralPipeline):
        return 2
    # elif clazz == type(AprilTagPipeline):
    #     return 3
    # elif clazz == type(CustomPipeline):
    #     return 4
    else:
        return -1