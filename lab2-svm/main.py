from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math

# Seed random to get same results each time
numpy.random.seed(100)

classA = [(random.normalvariate(-1.5,1),random.normalvariate(0.5,1),1.0)
          for i in range(5)] + \
          [(random.normalvariate(-1.5,1),random.normalvariate(1.5,1),1.0)
          for i in range(5)]
              
classB = [(random.normalvariate(-0.0,0.5),random.normalvariate(-0.5,0.5),-1.0)
          for i in range(10)]

def plot_data():
    pylab.plot([p[0] for p in classA],[p[1] for p in classA],'bo')
    pylab.plot([p[0] for p in classB],[p[1] for p in classB],'ro')
    
    pylab.show()

def create_model(data):
    print(data.shape)
    P = np.zeros(1,2)
    print(P)

def run():
    
    # put both classes in one vector
    data = classA + classB
    
    # mix up the vector
    random.shuffle(data)
    data = np.array(data)
    create_model(data)
    
    
    #plot_data()
    
if __name__ == "__main__":
    run()
