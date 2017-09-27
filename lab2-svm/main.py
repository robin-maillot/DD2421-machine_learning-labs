from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import pylab, random, math

def linearKernel(x, y):
	np.transpose(x)
	return x*y+1

def polyKernel(x, y, p)
	np.transpose(x)
	return math.pow((x*y+1),p)

def radialKernel(x, y, sigma)
	numer = -math.pow(x-y,2)
	denom = 2*math.pow(sigma,2)
	return math.exp(numer/denom)


