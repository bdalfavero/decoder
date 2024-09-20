from cuquantum import contract
from numpy.random import rand

a = rand(96, 64, 64, 96)
b = rand(96, 64, 64)
c = rand(64, 96, 64)

r = contract("mhkn,ukh,xuy->mxny", a, b, c)