import numpy as np
import math
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#----------------------------------------------------------------------------------------------
###1a
def Univariate(mean,var):
    
    std = math.sqrt(var)
    U = np.random.uniform(0,1)
    V = np.random.uniform(0,1)
    lnU = math.log(U)
    z = math.sqrt(-2*lnU)*math.cos(2*math.pi*V)
    answer = z*std + mean
    #print("The mean is %.4f . The std is %.4f . So the answer is %.4f"%(mean,std,answer))
    return answer
    
    """
    std = math.sqrt(var)
    random_data=np.sum(np.random.uniform(size = 12))-6
    random_data=random_data*std+mean
    return(random_data)
    """     
mean =  10
var  =   2    
#Univariate(mean,var)

#----------------------------------------------------------------------------------------------
###1b
def Polynomial(n,a,w,Univariate):
    e = Univariate(0,a)
    x = np.random.uniform(-1,1)
    phi_x = np.array([x ** i for i in range(n)])
    y = np.sum(phi_x * w) + e
    return x,y
n = 3
a = 1
w = np.random.rand(1,n)
#x,y = Polynomial(n,a,w,Univariate)


#------------------------------------------------------------------------------------------------
###2
def mean_data(data):
    return sum(data)/len(data)   
def variance_data(data):
    m = mean_data(data)
    x = [ (i-m)**2 for i in data]
    return sum(x)/(len(data)-1)
def change_variance(data,old_var,old_m,new_m,count):
    new_var = old_var + ((data[count-1]-old_m)**2)/len(data)-old_var/(len(data)-1)
    return new_var
def Sequential_Estimator(mean,var,Univariate,mean_data,variance_data):
    threshold = 0.0005
    data = []
    data.append(Univariate(mean,var))
    new_m = mean_data(data)
    new_var = 0
    print("Add data :%.10f"%(data[-1]),end="  ")
    print("Mean:%.10f  Var:0"%new_m)
    count = 1
    #for i in range(20000):
    while(True):
        count+=1
        data.append(Univariate(mean,var))
        print("The iteration is %d\t Add data :%.10f"%(count,data[count-1]),end="  ")
        old_m = new_m
        new_m = mean_data(data)
        old_var = new_var
        new_var = change_variance(data,old_var,old_m,new_m,count)
        print("Mean:%.10f  Var:%.10f"%(new_m,new_var))
        if (abs(new_m-old_m)<threshold) and (abs(new_var-old_var))<threshold:
            break
    return 0
mean = 3
var  = 5
#Sequential_Estimator(mean,var,Univariate,mean_data,variance_data)
#--------------------------------------------------------------------------------------------
##3
def matrix_print(matrix):
    N , M = matrix.shape
    for i in range(N):
        print("",end = "          ")
        for j in range(M):
            print(matrix[i][j],end="")
        print()
def baysian_linear_regression(b, n, a, w,Univariate):
    data, result = [], []
    
    mean = np.zeros((n, 1))
    cov = np.identity(n) * 1 / b
    #Prior(w) ~ N(0,b^(-1)*I)
    
    for i in range(10000):
        x , y = Polynomial(n,a,w,Univariate)
        data.append([x,y])
        print("-----------------------------------")
        print("Add data point %.10f %.10f"%(x,y))
        phi_x = np.array([[x**i for i in range(n)]])
        
        cov_new = np.linalg.inv( a * np.dot(phi_x.T , phi_x) + np.linalg.inv(cov))
        mean = np.dot(cov_new , a*phi_x.T * y +np.dot(np.linalg.inv(cov),mean))
        cov  = cov_new
        
        predictive_mean = np.dot(phi_x,mean).squeeze()
        predictive_var = 1/a + np.dot(np.dot(phi_x,cov),phi_x.T).squeeze()
        
        if i==9 or i==49:
            result.append([mean,cov])
        
        print("Postiror mean :")
        matrix_print(mean)
        print("Postiror variance :")
        matrix_print(cov)
        print("Predictive distribution ~N(%.5f , %.5f)"%(predictive_mean,predictive_var))
    
    result_title = ["Predict result", "After 10 incomes", "After 50 incomes"]
    point_amount = [-1, 10, 50]
    
    plot_n = 100
    data = np.array(data)
    result.insert(0,[mean,cov])

    plot_x = np.array(range(plot_n)) / plot_n * 4 - 2
    plot_phi_x = np.array([[p ** i for i in range(n)] for p in plot_x])
    plot_mean = np.array([np.dot(w, plot_phi_x[i]) for i in range(plot_n)])
    

    
    plt.subplot(2, 2, 1)
    plt.title("Ground truth")
    plt.axis([-2, 2, -20, 25])
    
    plt.plot(plot_x, plot_mean, 'b')
    plt.plot(plot_x, plot_mean + 1 / a, 'r')
    plt.plot(plot_x, plot_mean - 1 / a, 'r')
    
    for i in range(3):
        plot_mean = np.array([np.dot(result[i][0].T, plot_phi_x[j]) for j in range(plot_n)])
        plot_var = np.array([[1/a + np.dot(np.dot(plot_phi_x[j], result[i][1]), plot_phi_x[j].T)] for j in range(plot_n)])
        
        plt.subplot(2, 2, i + 2)
        plt.title(result_title[i])
        plt.axis([-2, 2, -20, 25])
        
        plt.scatter(data[:, 0][:point_amount[i]], data[:, 1][:point_amount[i]], s=10)
        plt.plot(plot_x, plot_mean, 'b')
        plt.plot(plot_x, plot_mean + plot_var, 'r')
        plt.plot(plot_x, plot_mean - plot_var, 'r')
    plt.savefig("hw3.png")
b=100
n=4
a=1
w=np.array([1,2,3,4])
baysian_linear_regression(b, n, a, w,Univariate)