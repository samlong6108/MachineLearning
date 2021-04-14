import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
import struct
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")
#------------------------------------------------------------------
def generate(mean,variance,number):
    answer = np.random.normal(mean,variance,size=(number))
    return answer
#------------------------------------------------------------------    
def plot(N,mx1,my1,mx2,my2,vx1,vx2,vy2,D1x,D1y,D2x,D2y,w,w1,ground_truth_matrix,design_matrix):
    plt.subplot(1,3,1)
    plt.plot(D1x,D1y,'o',color ='b')
    plt.plot(D2x,D2y,'o',color ='r')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title("Ground_Truth")
    
    plt.subplot(1,3,2)
    plt.title("Gradient descent")
    for i in range(N*2):
        if (1/(1+np.exp(-np.dot(design_matrix[i],w))))>0.5 :
            plt.plot(design_matrix[i][0],design_matrix[i][1],'o',color = 'r')
        else:
            plt.plot(design_matrix[i][0],design_matrix[i][1],'o',color = 'b')
    
    plt.subplot(1,3,3)
    plt.title("Newton's method")
    for i in range(N*2):
        if (1/(1+np.exp(-np.dot(design_matrix[i],w1))))>0.5 :
            plt.plot(design_matrix[i][0],design_matrix[i][1],'o',color = 'r')
        else:
            plt.plot(design_matrix[i][0],design_matrix[i][1],'o',color = 'b')

    
    
    plt.savefig("hw4.png")
#------------------------------------------------------------------
def generate_design_matrix(D1x,D1y,D2x,D2y):
    design_matrix = np.zeros((len(D1x)*2,3))
    design_matrix[0:len(D1x),0] = D1x
    design_matrix[len(D1x):,0 ] = D2x
    design_matrix[0:len(D1x),1] = D1y
    design_matrix[len(D1x):,1] = D2y
    design_matrix[:,2] = 1
    
    ground_truth_matrix = np.zeros((len(D1x)*2,1))
    ground_truth_matrix[len(D1x):]=1
    
    return design_matrix,ground_truth_matrix
#------------------------------------------------------------------
def logistic_Gradient_descent(design_matrix,ground_truth_matrix,generate,accuracy):
    w   = generate(0,1,3)
    w   = w.reshape(3,1)
    #for i in range(2000):
    while(True):
        old_w = w
        temp =( 1 / (1+np.exp(-1*np.dot(design_matrix,w)))) > 0.5
        temp = ground_truth_matrix- temp
        #print(temp)
        w =  w +  0.1*np.dot(design_matrix.transpose(),temp)
        #print(w,"\n")
        if ((old_w-w<0.1).all()):
            print("-----------------------------------------------")
            print("\t\tGradient descent:")
            print("w:")
            print(w,"\n")
            accuracy(w,ground_truth_matrix,design_matrix)
            return w   
            break
#------------------------------------------------------------------
def logistic_Newton(design_matrix,ground_truth_matrix,generate):
    print("---------------------------------------------------------")
    w = generate(0,1,3)
    w = w.reshape(3,1)    
    while(True):
        old_w = w
        temp =  1 / (1+np.exp(-1*np.dot(design_matrix,w)))
        temp = ground_truth_matrix- temp
        temp = np.dot(design_matrix.T,temp)
        #Generate D
        D = np.zeros((len(design_matrix),len(design_matrix)))
        for i in range(len(design_matrix)):
            temp1 = np.exp(-1*np.dot(design_matrix[i],w))
            if math.isinf(temp1):
                temp1 = np.exp(30)
            temp2 = (1+temp1)**2
            D[i,i] = float(temp1)/float(temp2)
        #Generate Hession matrix
        H = np.dot(np.dot(design_matrix.T,D),design_matrix)
        #Calculate new w
        if(np.linalg.det(H)==0):
            w = w + 0.1*temp
        else:
            w = w + np.dot(np.linalg.inv(H),temp)
        if (old_w - w<0.1).all() :
            print("\t\tNewton's method:")
            print("w:")
            print(w,"\n")
            accuracy(w,ground_truth_matrix,design_matrix)
            return w   
            break            
#------------------------------------------------------------------
def accuracy(w,gorund_truth_matrix,design_matrix):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(ground_truth_matrix)):
        predict  = (1/(1+np.exp(-np.dot(design_matrix[i],w))))>0.5
        if ground_truth_matrix[i] == 0:
            if (ground_truth_matrix[i]- predict) == 0 :
                TN = TN+1
            else :
                FP = FP+1
        if ground_truth_matrix[i] == 1 :
            if (ground_truth_matrix[i]- predict) == 0 :
                TP = TP+1
            else:
                FN = FN+1                
    print("Confusion Matrix:")
    print("\t\t\tIs cluster 2\tIS cluster 1")
    print("Predict cluster 2\t%d\t\t%d"%(TP,FP))
    print("Predict cluster 1\t%d\t\t%d"%(FN,TN))
    print("Sensitivity is %f"%(TP/(TP+FN)))
    print("Specificity is %f"%(TN/(FP+TN)))
#------------------------------------------------------------------ 
"""          
N = 50
mx1 = 1 
my1 = 1
mx2 = 3
my2 = 3
vx1 = 2
vy1 = 2
vx2 = 4
vy2 = 4
D1x = generate(mx1,vx1,N)
D1y = generate(my1,vy1,N)
D2x = generate(mx2,vx2,N)
D2y = generate(my2,vy2,N)

design_matrix,ground_truth_matrix = generate_design_matrix(D1x,D1y,D2x,D2y)
w =logistic_Gradient_descent(design_matrix,ground_truth_matrix,generate,accuracy)
w1 = logistic_Newton(design_matrix,ground_truth_matrix,generate)
plot(N,mx1,my1,mx2,my2,vx1,vx2,vy2,D1x,D1y,D2x,D2y,w,w1,ground_truth_matrix,design_matrix)
"""
#------------------------------------------------------------------------
#------------------------------EM algorithm------------------------------    
def loadimages(train_images):
    print("--------------------------------------------------")
    print("Now on ",train_images)
    train_images = open(train_images, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    #Represent BIG ENDIAN   I represent INTEGER
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, train_images, offset)
    print("Magic:%d, Number of image: %d, Image size: %d*%d"% (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset is ",offset)

    fmt_image = '>' + str(image_size) + 'B'
    print("fmt_image is " ,fmt_image)

    images = np.empty((num_images, num_rows, num_cols))
    print("images shape is " ,images.shape)

    for i in range(num_images):
        if (i + 1) % num_images == 0:
            print('Already  %d' % (i + 1) + 'pictures')
        images[i] = np.array(struct.unpack_from(fmt_image, train_images, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images
#------------------------------------------------------------------------
def loadlabel(train_labels):
    print("---------------------------------")
    print("Now on",train_labels)
    bin_data = open(train_labels, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('Magic:%d, Number of images: %d' % (magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    print("fmt_image is ",struct.calcsize(fmt_image))
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % num_images == 0:
            print('Already %d' % (i + 1) + 'pictures')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels
#------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def update_posterior(X_train , Lambda , Distribution):
    '''
    update posterior using log likelihood
    :param X_train: (60000,784) 0-1 uint8 matrix
    :param Lambda: (10,1)
    :param Distribution: (10,784)
    :return: (60000,10)
    '''
    Distribution_complement = 1 - Distribution
    w = np.zeros((60000,10))
    for i in range(60000):
        for j in range(10):
            w[i,j] = np.prod(X_train[i] * Distribution[j] + (1-X_train[i])*Distribution_complement[j])
            
        
    #Add prior
    w = w*Lambda.reshape(1,-1)
    
    #normalized each row to [0,1] % sum = 1
    sums = np.sum(w,axis = 1).reshape(-1,1)
    sums[sums==0] = 1
    w = w/sums
    return w
#----------------------------------------------------------------------------------------------------------------    
def update_lambda(w):
    '''
    :param W: (60000,10)
    :return: (10,1)
    '''
    L = np.sum(w,axis=0)
    L = L/60000
    return L.T
#----------------------------------------------------------------------------------------------------------------
def update_distribution(A,w):
    '''
    A.T@W -> normalized,transpose -> concate with 1-complement
    :param A: (60000,784)
    :param W: (60000,10)
    :return: (10,784)
    '''
    #normalized W
    sums = np.sum(w,axis = 0)
    sums[sums==0] = 1
    w_noramlized = w/sums
    P = np.dot(A.T,w_noramlized)
    return P.T
#----------------------------------------------------------------------------------------------------------------
def get_pixvalueProb_discrete(train_x,train_y):
    '''
    get pixvalue_prob conditional on class & dim
    :param train_x: (60000,784) 0-1 matrix
    :param train_y: (60000,)
    :return: (10,784) probability matrix of pixelValue==1
    '''
    labels = np.zeros(10)
    for label in train_y:
        label = int(label)
        labels[label] +=1
    
    distribution = np.zeros((10,784))
    for i in range(60000):
        c = train_y[i]
        for j in range(784):
            if train_x[i][j]==1:
                c = int(c)
                distribution[c][j] +=1
    
    distribution = distribution /labels.reshape(-1,1)
    
    return distribution
#----------------------------------------------------------------------------------------------------------------
def plot_discrete(Distribution):
    '''
    :param Distribution: (10,784)
    :return:
    '''
    for c in range(10):
        print('class',c)
        for i in range(28):
            for j in range(28):
                print(1 if Distribution[c,i*28+j]>0.5 else 0,end=' ')
            print()
        print()
        print()
#----------------------------------------------------------------------------------------------------------------
def distance(a,b):
    '''
    :param a: (784)
    :param b: (784)
    :return: euclidean distance between a and b
    '''
    return np.linalg.norm(a-b)
#----------------------------------------------------------------------------------------------------------------
def hungarian_algo(Cost):
    '''
    match GT to our estimate
    :param Cost: (10,10)
    :return: (10) column index
    '''
    row_idx,col_idx=linear_sum_assignment(Cost)
    return col_idx
#----------------------------------------------------------------------------------------------------------------
def perfect_matching(ground_truth,estimate,distance,humgarian_algo):
    '''
    matching GT_distribution to estimate_distribution by minimizing the sum of distance
    :param ground_truth: (10,784)
    :param estimate: (10,784)
    :return: (10)
    '''
    Cost = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            Cost[i,j] = distance(ground_truth[i],estimate[j])
    classes_order = humgarian_algo(Cost)
    return classes_order    
#----------------------------------------------------------------------------------------------------------------
def plot(Distribution,classes_order,threshold):
    '''
    plot each classes expected pattern
    :param Distribution: (10,784)
    :param classes_order: (10)
    :param threshold: value between 0.0~1.0
    :return:
    '''
    Pattern=np.asarray(Distribution>threshold,dtype='uint8')
    for i in range(10):
        print('class {}:'.format(i))
        plot_pattern(Pattern[classes_order[i]])
    return
#----------------------------------------------------------------------------------------------------------------
def confusion_matrix(real,predict,classes_order):
    '''
    :param real: (60000)
    :param predict: (60000)
    :param classes_order: (10)
    :return:
    '''
    for i in range(10):
        c=classes_order[i]
        TP,FN,FP,TN=0,0,0,0
        for i in range(60000):
            if real[i]!=c and predict[i]!=c:
                TN+=1
            elif real[i]==c and predict[i]==c:
                TP+=1
            elif real[i]!=c and predict[i]==c:
                FP+=1
            else:
                FN+=1
        plot_confusion_matrix(c,TP,FN,FP,TN) 
#----------------------------------------------------------------------------------------------------------------
def print_error_rate(count,real,predict,classes_order):
    '''
    :param count: int
    :param real: (60000)
    :param predict: (60000)
    :param classes_order: (10)
    :return:
    '''
    print('Total iteration to converge: {}'.format(count))
    real_transform=np.zeros(60000)
    for i in range(60000):
        j = int(real[i])
        real_transform[i]=classes_order[j]
    error=np.count_nonzero(real_transform-predict)
    print('Total error rate: {}'.format(error/60000)) 
#----------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(c,TP,FN,FP,TN):
    print('------------------------------------------------------------')
    print()
    print('Confusion Matrix {}:'.format(c))
    print('\t\t\t  Predict number {} Predict not number {}'.format(c, c))
    print('Is number  \t{}\t\t{}\t\t\t\t{}'.format(c,TP,FN))
    print('Isn\'t number {}\t\t{}\t\t\t\t{}'.format(c,FP,TN))
    print()
    print('Sensitivity (Successfully predict number {}    ): {:.5f}'.format(c,TP/(TP+FN)))
    print('Specificity (Successfully predict not number {}): {:.5f}'.format(c,TN/(TN+FP)))
    print()
 
#----------------------------------------------------------------------------------------------------------------
def plot_pattern(pattern):
    '''
    :param pattern: (784)
    :return:
    '''
    for i in range(28):
        for j in range(28):
            print(pattern[i*28+j],end=' ')
        print()
    print()
    print()
    return       
#---------------------------------------------------------------------------------------------------------------- 
train_images = './train-images.idx3-ubyte'
train_labels = './train-labels.idx1-ubyte' 
train_images = loadimages(train_images)
train_labels = loadlabel(train_labels)
train_images = train_images.reshape(60000,784)
for i in range(60000):
    for j in range(784):
        if train_images[i][j]>=128:
            train_images[i][j] = 1
        else:
            train_images[i][j]= 0

eps = 1
L = np.random.rand(10)
L = L/np.sum(L)
P = np.random.rand(10,784)

last_diff , diff, count = 1000,100,0
while abs(last_diff-diff)>eps and diff>eps:

    #E step
    w = update_posterior(train_images,L,P)
    
    #M step
    L_new=update_lambda(w)
    P_new=update_distribution(train_images,w)
    
    last_diff=diff
    diff=np.sum(np.abs(L-L_new))+np.sum(np.abs(P-P_new))
    print('diff: ',diff)
    print('Lambda:',L_new.reshape(1,-1)[0])
    L=L_new
    P=P_new
    count+=1    
    
maxs=np.argmax(w,axis=1)
unique,counts=np.unique(maxs,return_counts=True)
print(dict(zip(unique,counts)))
print('Lambda:',L.reshape(1,-1))

#plot classes predict & confusion matrix
GT_distribution=get_pixvalueProb_discrete(train_images,train_labels)

#plot_discrete(P)

class_order=perfect_matching(GT_distribution,P,distance,hungarian_algo)
plot(P,class_order,threshold=0.35)
confusion_matrix(train_labels,maxs,class_order)
print_error_rate(count,train_labels,maxs,class_order)