from cvxopt.solvers import qp
from cvxopt.base import matrix
import matplotlib
matplotlib.use('Agg')
import pylab, random, math
import numpy as np

# Seed random to get same results each time
#random.seed(100)

g, axarr = pylab.subplots(2, 2)

classA = [(random.normalvariate(-1.5,1),random.normalvariate(0.5,1),1.0)
          for i in range(5)] + \
          [(random.normalvariate(-1.5,1),random.normalvariate(1.5,1),1.0)
          for i in range(5)]
              
classB = [(random.normalvariate(-0.0,0.5),random.normalvariate(-4,4),-1.0)
          for i in range(10)]


def plot_data(f,m):
    
    xrange = np.arange(-4,4,0.05)
    yrange = np.arange(-4,4,0.05)
    
    grid = matrix([[indicator(f,[x,y],m) for y in yrange] for x in xrange])
    axarr[m/2,m%2].contour(xrange,yrange,grid,(-1.0,0.0,1.0),colors=('red','black','blue'),linewidths=(1,3,1))
    axarr[m/2,m%2].plot([p[0] for p in classA],[p[1] for p in classA],'bo')
    axarr[m/2,m%2].plot([p[0] for p in classB],[p[1] for p in classB],'ro')   

    
def linearKernel(x, y):
    np.transpose(x)
    return (np.dot(x,y)+1)

def polyKernel(x, y, p=2):
	np.transpose(x)
	return math.pow((np.dot(x,y)+1),p)

def radialKernel(x, y, sigma=1):
	numer = -np.dot(x-y,x-y)
	denom = 2*math.pow(sigma,2)
	return math.exp(numer/denom)
    
def filter(alpha,data, C):
    ind = np.array([0,0,0,0])
    # avg = 0
    # for i in range(len(alpha)):
    #     if(alpha[i]>000001):
    #         avg += alpha[i]
    # avg /= (i+1)
    for i in range(len(alpha)):
        #if(alpha[i]>0.000001 and alpha[i] < 5.0*avg):
        if(alpha[i]>0.000001 and alpha[i] < C):
            ind = np.vstack((ind,[data[i,0],data[i,1],data[i,2],alpha[i]]))
    try:
    	return ind[1:,:], C
    except:
		print ("ERROR: C value too low, increasing to %d \n" %(C))
		return filter(alpha, data, C*10)


def indicator(ind,new_data,m):
    ans = 0
    for i in range(ind.shape[0]):
        if(m==0):
            ans += linearKernel(new_data, ind[i][[0,1]])*ind[i][2]*ind[i][3]
        if(m==1):
            ans += polyKernel(new_data, ind[i][[0,1]],2)*ind[i][2]*ind[i][3]
        if(m==2):
            ans += polyKernel(new_data, ind[i][[0,1]],3)*ind[i][2]*ind[i][3]
        if(m==3):
            ans += radialKernel(new_data, ind[i][[0,1]])*ind[i][2]*ind[i][3]
    return ans

def create_model(data,m):
    n = data.shape[0]
    P = np.zeros((n,n))
    
    if(m==0):
        for i in range(n):
            for j in range(n):
                P[i][j] = data[i][2]*data[j][2]*linearKernel(data[i][[0,1]], data[j][[0,1]])
    if(m==1):
        for i in range(n):
            for j in range(n):
                P[i][j] = data[i][2]*data[j][2]*polyKernel(data[i][[0,1]], data[j][[0,1]],2)
    if(m==2):
        for i in range(n):
            for j in range(n):
                P[i][j] = data[i][2]*data[j][2]*polyKernel(data[i][[0,1]], data[j][[0,1]],3)
    if(m==3):
        for i in range(n):
            for j in range(n):
                P[i][j] = data[i][2]*data[j][2]*radialKernel(data[i][[0,1]], data[j][[0,1]])

    q = np.zeros((n,1)) -1
    #print(q)
    #G = np.zeros(n,1)
    G = np.zeros((2*n,n))
    for i in range(n):
    	G[i, i] = -1

    
    h = np.zeros((2*n,1))
    #print(h)
    
    return P,q,G,h

def run():
    # put both classes in one vector
    data = classA + classB
    
    # mix up the vector
    random.shuffle(data)
    data = np.array(data)
    C = []

    for i in range(4):
        P,q,G,h = create_model(data,i)
        r = qp(matrix(P),matrix(q),matrix(G),matrix(h))
        alpha = list(r['x'])
    
        #print(alpha)
        f, tmp = filter(alpha, data, 100)
        C.append(tmp)
        print C[i]
        
        plot_data(f,i)
    axarr[0, 0].set_title('Linear, C=' + str(C[0]))
    axarr[0, 1].set_title('2nd order Poly, C=' + str(C[1]))
    axarr[1, 0].set_title('3rd order Poly, C=' + str(C[2]))
    axarr[1, 1].set_title('Gaussian, C=' + str(C[3]))
    pylab.show()
    pylab.savefig("HyperDerp.png")

if __name__ == "__main__":
    run()
