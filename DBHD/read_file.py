import numpy as np 


def read_file(str,i): 
    dic = {} 
    classmember = 0 
    if (i == 0): 
        file = open("data" + "/" + "artificial" + "/" + str + ".arff", "r") 
    elif i==1: 
        file = open("data" + "/" + "real-world" + "/" + str + ".arff", "r") 
    else: 
        #ucipp-master/uci/abalone-3class.arff 
        file = open("ucipp-master/uci/" + str + ".arff", "r") 
    y = [] 
    label = [] 
    for line in file: 
        if (line.startswith("@") or line.startswith("%") or len(line.strip()) == 0): 
            pass 
        else: 
            j = line.split(",") 
            alpha = 1 
            if ("?" in j): 
                continue 
            x = 0 
            k = [] 
            for i in range(len(j) - 1): 
                k.append(float(j[i]) * alpha + x) 
            if (not j[len(j) - 1].startswith("noise")): 
                str = j[len(j) - 1].rstrip() 
                if(str in dic.keys()): 
                    label.append(dic[str]) 
                else: 
                    dic[str]= classmember 
                    label.append(dic[str]) 
                    classmember +=1 
                    ppppp = 9 
            else: 
                label.append(-1) 
            y.append(k) 
    #print(y) 
    #print(np.array(y)) 
    return np.array(y), np.array(label).reshape(1, len(label))[0] 



#print(read_file('diamond9', 0)) 
#read_wdbc('wdbc', 1) 
'''
X, Y = read_file('cluto-t4-8k', 0) 
print(X) 
print(Y)
'''
