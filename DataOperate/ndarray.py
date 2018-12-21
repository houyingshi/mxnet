from mxnet import nd
import numpy as np

X = nd.arange(12).reshape((3, 4))
print(X)

P = np.ones((2, 3))
print(P)
D = nd.array(P)
print(D)

print(D.asnumpy())