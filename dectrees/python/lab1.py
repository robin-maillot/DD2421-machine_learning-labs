import dtree as d
import monkdata as m
import drawtree_qt5 as qt5
    
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
    print(t)
    print(float(pCount/len(dataset)))
    return pCount
    
def test(dataset):
    max = 0.0
    split_attribute = 0
    for i in range(0,len(m.attributes)):
        if(d.averageGain(dataset,m.attributes[i])>max):
            max = d.averageGain(dataset,m.attributes[i])
            split_attribute = i
            
    acc = 0.0
    for v in m.attributes[split_attribute].values:
        subset = d.select(dataset, m.attributes[split_attribute], v)
        x = get_results(subset)
        acc += x
    acc = acc/len(dataset)
    return acc

## Question 1 ##
print("Question 1:\n")
get_entropy()

## Question 2 ##
print("Question 2:\n")
get_gain(m.monk1,"MONK-1")
get_gain(m.monk2,"MONK-2")
get_gain(m.monk3,"MONK-3")

## Question 3 ##
print("Question 3:\n")
split_set(m.monk1, m.attributes[4], "MONK-1")
get_results(m.monk1)

acc = test(m.monk1)
print("Accuracy after one step: " + str(acc))

t=d.buildTree(m.monk1, m.attributes,2);
#print(d.check(t, m.monk1test))
#qt5.drawTree(t)