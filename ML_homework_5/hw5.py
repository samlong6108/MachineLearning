import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
#--------------------------------------------------------------------------------------
def dataloader(path):
    x = []
    y = []
    f = open(path,'r')
    for line in f.readlines():
        data = line.split(' ')
        x.append(float(data[0]))
        y.append(float(data[1]))
    x = np.array(x)
    y = np.array(y)
    return x,y

#--------------------------------------------------------------------------------------
def cal_kernel(x1,x2,alpha,length_scale):
    '''
    using rational quadratic kernel function: k(x_i, x_j) = (1 + (x_i-x_j)^2 / (2*alpha * length_scale^2))^-alpha
    :param X1: (n) ndarray
    :param X2: (m) ndarray
    return: (n,m)  ndarray
    '''
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(1,-1)
    temp = np.power(x1-x2,2)
    kernel = np.power(1+temp/(2*alpha-length_scale**2),-alpha)
    return kernel
#--------------------------------------------------------------------------------------
def predict(x_line , X, y ,K ,beta, alpha , length_scale,cal_kernel):
    '''
    vectorize calculate k_x_xstar !!
    :param x_line: sampling in linspace(-60,60)
    :param X:  (n) ndarray
    :param y: (n) ndarray
    :param K: (n,n) ndarray
    :param beta:
    :return: (len(x_line),1) ndarray, (len(x_line),len(x_line)) ndarray
    '''    
    k_x_xstar = cal_kernel(X,x_line,alpha,length_scale)
    k_xstar_xstar = cal_kernel(x_line,x_line,alpha,length_scale)
    means = np.dot(np.dot(k_x_xstar.T,np.linalg.inv(K)),y.reshape(-1,1))
    k_star =  k_xstar_xstar+ (1/beta)
    variances = k_star - np.dot(np.dot(k_x_xstar.T,np.linalg.inv(K)),k_x_xstar)
    return means,variances
#--------------------------------------------------------------------------------------    
def plot_gaussian(x,y,x_line,mean_predict,variance_predict,name):
    plt.plot(x,y,'bo')
    plt.plot(x_line,mean_predict,'k-')
    plt.fill_between(x_line,mean_predict+2*variance_predict,mean_predict-2*variance_predict,facecolor='salmon')
    plt.xlim(-60,60)
    plt.savefig(name+".png")
#--------------------------------------------------------------------------------------   
def objective_function(X,y,beta):
    '''
    :param X:  (n) ndarray
    :param y:  (n) ndarray
    :param beta:
    :return:
    '''
    def objective(theta):
        K = cal_kernel(X,X,alpha=theta[0],length_scale=theta[1])+(1/beta)*np.identity(len(X))
        
        target1 = np.dot(np.dot(0.5*y.reshape(1,-1),np.linalg.inv(K)),y.reshape(-1,1))
        
        target2 = 0.5*np.log(np.linalg.det(K))
        
        target3 = 0.5*len(X)*np.log(2*np.pi)
        
        target = target1 + target2+target3 
        return target
        
    return objective
#----------------------------------------------------------------------------------------
def minimize_negative_margianl(X,y,beta):
    objective_value = 1e9
    inits= [1e-2,1e-1,0,1e1,1e2]
    for init_alpha in inits:
        for init_length_scale in inits:
            res=minimize(objective_function(X,y,beta),x0=[init_alpha,init_length_scale],bounds=((1e-5,1e5),(1e-5,1e5)))
            if res.fun<objective_value:
                objective_balue = res.fun
                alpha_optimize,length_scale_optimize=res.x
    print("Optimize alpha is %.3f Optimize length_scale is %.3f"%(alpha_optimize,length_scale_optimize))
    return alpha_optimize,length_scale_optimize

#----------------------------------------------------------------------------------------

###5.1
path = './input.data'
beta = 5
alpha = 1
length_scale = 1
name = "hw5"
x , y = dataloader(path)
kernel = cal_kernel(x,x,alpha,length_scale)+1/beta*np.identity(len(x))
x_line=np.linspace(-60,60,num=500)
mean_predict,variance_predict = predict(x_line , x, y ,kernel ,beta, alpha , length_scale,cal_kernel)
mean_predict=mean_predict.reshape(-1)
variance_predict = np.sqrt(np.diag(variance_predict))
plot_gaussian(x,y,x_line,mean_predict,variance_predict,name)


#----------------------------------------------------------------------------------------
###5.2
plt.clf()
path = './input.data'
beta = 5
name = "hw5_optimize"
x , y = dataloader(path)
alpha_optimize,length_scale_optimize =  minimize_negative_margianl(x,y,beta)
K=cal_kernel(x,x,alpha=alpha_optimize,length_scale=length_scale_optimize)+1/beta*np.identity(len(x))
x_line=np.linspace(-60,60,num=500)
mean_predict,variance_predict = predict(x_line , x, y ,K ,beta, alpha_optimize , length_scale_optimize,cal_kernel)
mean_predict=mean_predict.reshape(-1)
variance_predict = np.sqrt(np.diag(variance_predict))
plot_gaussian(x,y,x_line,mean_predict,variance_predict,name)