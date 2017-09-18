import dtree as d
import monkdata as m
import drawtree_qt5 as qt5
import random
import matplotlib.pyplot as plt
import numpy as np

    
def get_entropy():
    print("Entropy:\nMONK-1: " + str(d.entropy(m.monk1)) + "\n")
    print("MONK-2: " + str(d.entropy(m.monk2)) + "\n")
    print("MONK-3: " + str(d.entropy(m.monk3)) + "\n")
    return None

def get_gain(dataset,name):
    print("Information Gain for " + name)
    for i in range(0,len(m.attributes)):
        print("Attribute "+str(i)+" :"+str(d.averageGain(dataset,m.attributes[i])))
    print("\n")
    return None

def split_set(dataset,attribute,name):
    for v in attribute.values:
        subset = d.select(dataset, attribute, v)
        get_gain(subset,name+"-"+str(v))
    return None

def get_results(dataset):
    t = d.mostCommon(dataset)
    pCount = len([x for x in dataset if (x.positive==t)])
    #print(float(pCount/len(dataset)))
    return pCount
    
def test(train,test):
    max = 0.0
    split_attribute = 0
    for i in range(0,len(m.attributes)):
        if(d.averageGain(train,m.attributes[i])>max):
            max = d.averageGain(train,m.attributes[i])
            split_attribute = i
            
    acc = 0.0
    for v in m.attributes[split_attribute].values:
        max2 = 0.0
        split_attribute2 = 0
        train_subset = d.select(train, m.attributes[split_attribute], v)
        test_subset = d.select(test, m.attributes[split_attribute], v)
        for i in range(0,len(m.attributes)):
            if(d.averageGain(train_subset,m.attributes[i])>max):
                max = d.averageGain(train_subset,m.attributes[i])
                split_attribute2 = i
                
                
        for v in m.attributes[split_attribute].values:
            test_subset2 = d.select(test_subset, m.attributes[split_attribute2], v)
            x = get_results(test_subset2)
            acc += x
    acc = acc/len(test)
    return acc

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruneTree(tree,val):
    max = d.check(tree,val);
    best_prunned_tree = tree
    it = 0
    while True:
        max2 = 0
        best_i = -1
        pruned_trees = d.allPruned(best_prunned_tree)
        for i in range(0,len(pruned_trees)):
            x = d.check(pruned_trees[i],val)
            if(x>max2):
                max2 = x
                best_i = i
        #print(str(it) + " " + str(max) + " " + str(max2)+" " + str(best_i))
        if(max2>=max):
            if(best_i==-1):
                break
            best_prunned_tree = pruned_trees[best_i]
            max = max2
        else:
            break
        it += 1
  
    return best_prunned_tree

## Question 1 ##
print("Question 1:\n")
get_entropy()

## Question 2 ##
print("Part 3: Entropy\n")
get_gain(m.monk1,"MONK-1")
get_gain(m.monk2,"MONK-2")
get_gain(m.monk3,"MONK-3")

## Question 3 ##
print("Part 4: Information Gain\n")
split_set(m.monk1, m.attributes[4], "MONK-1")
get_results(m.monk1)


## Question 4 ##
print("Part 5: Building Decision Trees\n")
acc = test(m.monk1,m.monk1test)
print("Accuracy at depth 2: " + str(acc))

t=d.buildTree(m.monk1, m.attributes,2);
print("Accuracy using ID3: " + str(d.check(t, m.monk1test)))
#qt5.drawTree(t)

## Question 5 ##
print("Part 6: Prunning\n")

monk1train, monk1val = partition(m.monk1, 0.6)
t=d.buildTree(monk1train, m.attributes)
prunned_t = pruneTree(t,monk1val)
print("Accuracy before prunning: " + str(d.check(t, m.monk1test)))
print("Accuracy after prunning: " + str(d.check(prunned_t, m.monk1test)))
#qt5.drawTree(prunned_t)


## last question ##
trainval = m.monk3
test = m.monk3test
fractions = [0.3,0.4,0.5,0.6,0.7,0.8]
runs = 100
x = []
y = []
means = []
vars = []
for f in fractions:
    xf = []
    yf = []
    for run in range(0,runs):
        train, val = partition(trainval,f)
        t=d.buildTree(train, m.attributes)
        prunned_t = pruneTree(t,val)
        yf.append(d.check(prunned_t, test)) 
        xf.append(f)
    means.append(np.mean(yf))
    vars.append(np.var(yf))
    x.append(xf)
    y.append(yf)

means = np.array(means)
vars = np.array(vars)
plt.plot(x,y, 'bo',fractions,means,'ro',fractions,means+vars,'ro',fractions,means-vars,'ro')
plt.ylabel('test error')
plt.ylabel('fraction of data used for training')

plt.show()

