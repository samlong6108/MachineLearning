import numpy as np
import struct
import matplotlib.pyplot as plt
import math 

train_images = './train-images.idx3-ubyte'
train_labels = './train-labels.idx1-ubyte'
test_images = './t10k-images.idx3-ubyte'
test_labels = './t10k-labels.idx1-ubyte'



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
train_images = loadimages(train_images)
train_labels = loadlabel(train_labels)
test_images  = loadimages(test_images)
test_labels  = loadlabel(test_labels)
print("--------- prior------------")
prior=[0]*10 
num_of_label=[0]*10
for i in range(len(train_labels)):
    prior[int(train_labels[i])]+=1
    num_of_label[int(train_labels[i])]+=1
_sum=0
for i in range(len(prior)):
    prior[i]/=len(train_labels)
prior=np.array(prior)
print(prior)

#-----------------------------------------------------------------------------------------------------------

train_images = train_images.reshape(60000,784)
test_images  = test_images.reshape(10000,784) 
mean = np.zeros((10,28*28),dtype = np.float64)
std  = np.zeros((10,28*28),dtype = np.float64)
for index , label in enumerate(np.unique(train_labels)):   #np.unique sort and remove repeat numbers
    label_index = np.where(label==train_labels)
    mean[index] = np.mean(train_images[label_index],axis=0)
    std[index]  = np.std(train_images[label_index],axis =0)
print(std[0].shape)
std = std + 20
print(np.min(std))
#-----------------------------------------------------------------------------------------------------------
def continous(test_images,mean,std,test_labels):
    log_p_test = np.zeros(10, dtype = 'float64')
    count = 0
    for k in range(10000):
        log_likelihood=[0]*10
        log_likelihood=np.array(log_likelihood,dtype=float)
        for i in range(10):
            temp = 0
            log_p_test[i] = (prior[i])
            for j in range(784):
                if std[i][j]!=0:
                   
                    #log_p_test[i] += -1/2*np.log(std[i,j]*2*math.pi) -1/2*((test_images[k, j] - mean[i,j]) ** 2) / std[i,j]
                   
                
                    log_inside=float(2*(math.pi)*(std[i][j]**2))
                    log_inside=math.sqrt(log_inside)
                    first_part=float(math.log(log_inside))
                    second_part=float(((test_images[k][j]-mean[i][j])**2)/(2*(std[i][j])))
                    temp=float(temp+first_part+second_part)
                    log_p_test[i] = temp + math.log(prior[i])
                
                    #exp_term = (-1/2)*(test_images[k][j]-mean[i][j])**2/(std[i][j]**2)
                    #first_term = math.log(math.sqrt(1/(2*(math.pi)*std[i][j]**2)))
                    #temp = first_term + exp_term +temp
            temp = temp + math.log(prior[i])
            temp = temp
            log_likelihood[i]=temp  
        predict = (np.argmin(log_likelihood))
        result = test_labels[k]
        normalize_sum=0
        
        for i in range(10):
            normalize_sum+=log_likelihood[i]
        for i in range(10):
            log_likelihood[i]=float(log_likelihood[i]/normalize_sum)
        
        print("-----------------------")
        print("image[%d]"%k)
        for i in range(len(log_likelihood)):
            print('%d   %f'%(i,log_likelihood[i]))
        
        
        print("predict is",(np.argmin(log_likelihood)))
        print("Result is %d"%test_labels[k])
        if predict == result :
            count =count+1
        #break
    accuracy = float(count/10000)
    error    = 1-accuracy
    print("Accuracy is %f"%accuracy)
    print("Error is %f"%error)
    
 
#-----------------------------------------------------------------------------------------------------------
def discrete(test_images,mean,std,test_labels,train_labels,train_images,num_of_label,prior):
    count = 0
    print("Start to calculate training data")
    train_discrete_images=np.zeros((10,784,32),dtype=np.float64)
    for i in range(60000):
        label_num=int(train_labels[i])
        for j in range(784):
            class_of_pixel=int(train_images[i][j]/8)
            train_discrete_images[label_num][j][class_of_pixel]+=1
    
    for i in range(10):
        train_discrete_images[i]=train_discrete_images[i]/int(num_of_label[i])
    print("-----------------Here-------------------------")
    print(train_discrete_images[:, 0, 0:16])
    print("-----------------Here-------------------------")
    print("Start to calculate testing data")
    
    for im in range(10000):
        likelihood=[0]*10
        likelihood=np.array(likelihood,dtype=np.float64)
        for i in range(10):
            temp=0
            for j in range(784):
                now_bin=int(test_images[im][j]/8)
                temp+=np.log(train_discrete_images[i][j][now_bin]) +0.001
                #print("The iteration is %d %.70f"%(j,temp))
            temp+=np.log(prior[i])
            likelihood[i]=temp        
        #for i in range(len(likelihood)):
        #    print("%.50f"%likelihood[i])
        predict = (np.argmax(likelihood))
        result = test_labels[im]  
        normalize_sum=0  
        for i in range(10):
            normalize_sum+=likelihood[i]
        for i in range(10):
            likelihood[i]=float(likelihood[i]/normalize_sum)
        
                  
        
        print("-----------------------")
        print("image[%d]"%im)
        for i in range(len(likelihood)):
            print('%d   %f'%(i,likelihood[i]))
        
        
        print("predict is %d"%predict)
        print("Result is %d"%test_labels[im])
        if predict == result :
            count =count+1
        #break
    accuracy = float(count/10000)
    error    = 1-accuracy
    print("Accuracy is %f"%accuracy)
    print("Error is %f"%error)
    
    print("Imagination of numbers in Bayssian classifier:")
    for label in range(10):
        print("%d :"%(label))
        matrix = [0]*784
        for pixel in range(784):
            mean = 0
            for bin in range(32):
                mean += train_discrete_images[label][pixel][bin]*bin
            if pixel%28 == 0:
                print("")
            if mean <=15:
                print("0",end="")
            else:
                print("1",end="")
        #print("")
    """
    imagination = np.zeros((10, 784), dtype = 'int')

    for i in range(784):
        imagination[:, i] = np.sum(train_discrete_images[:, i, 0:16], axis = 1) < 1/2
    # print the imagination
    for i in range(10):
        print(str(i) + ':')
        for j in range(1, 784+1):
            print(imagination[i, j-1], sep = ' ', end = '')
            if j % 28 == 0:
                print()
    """
#continous(test_images,mean,std,test_labels)  
discrete(test_images,mean,std,test_labels,train_labels,train_images,num_of_label,prior)