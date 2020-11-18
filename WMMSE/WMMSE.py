import numpy as np
import random
import matplotlib.pyplot as plt

# Update of Uk, Wk, Vk
def update_Uk(A, U, V, H, K, Nr):
    for k in range(K):
        A_item = np.zeros((Nr,Nr),complex)
        for m in range(K):
            A_item += H[k].dot(V[m]).dot(V[m].conjugate().T).dot(H[k].conjugate().T)
        A[k] = sigma**2/PT*np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)])*np.eye(Nr) + A_item
        U[k] = np.linalg.inv(A[k]).dot(H[k]).dot(V[k])
    return U

def update_Wk(U, V, W_old, W, K, d):
    for k in range(K):
        E = np.eye(d) + 0j*np.eye(d) -(U[k].conjugate().T).dot(H[k]).dot(V[k])
        W_old[k] = W[k]
        W[k] = np.linalg.inv(E)
    return W, W_old

def update_Vk(V, W, U, H, K, Nt, sigma):
    for k in range(K):
        B_item = np.zeros((Nt,Nt),complex)
        for m in range(K):
            B_item += (H[m].conjugate().T).dot(U[m]).dot(W[m]).dot(U[m].conjugate().T).dot(H[m])
        B = B_item + sigma**2/PT*np.sum([np.trace(U[i].dot(W[i]).dot(U[i].conjugate().T)) for i in range(K)])*np.eye(Nt)
        V[k] = np.linalg.inv(B).dot(H[k].conjugate().T).dot(U[k]).dot(W[k])
    #Scale
    alpha = np.sqrt(PT/np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)]))
    for k in range(K):
        V[k] = alpha * V[k]   
    return V


K = 4
PT = 100
sigma = 1
epsilon = 1e-5
d = 2
Nr = 2
Nt = 8
Imax = 100  #最大迭代次数
err = float("inf")

rate=[]

for num_test in range(100):
    #initial H
    mean1 = np.zeros(Nr)
    cov1 = np.eye(Nr)
    data = np.zeros((Nr,Nt))+1j*np.zeros((Nr,Nt))
    H = np.zeros((K,Nr,Nt))+1j*np.zeros((K,Nr,Nt))
    for i in range(K):
        data1 = np.random.multivariate_normal(mean1,cov1,Nt)
        data2 = np.random.multivariate_normal(mean1,cov1,Nt)
        data[0,:] = data1[:,0]+1j*data1[:,1]
        data[1,:] = data2[:,0]+1j*data2[:,1]
        H[i,:,:] = data; 
    
    #initial A、U、W, W_old
    A = []
    U = []
    W = []
    V = []
    W_old = []
    for k in range(K):
        A.append(np.zeros((Nr,Nr),complex))
        U.append(np.zeros((Nr,d),complex))
        W.append(np.zeros((d,d),complex))
        W_old.append(np.eye(d)+complex("inf"))
        #V.append( np.sqrt(1/2)*(np.random.randn(Nt,Nr)+1j*np.random.randn(Nt,Nr)) )
        V.append( np.conj(np.matmul(np.linalg.pinv(np.matmul(H[k],np.conj(H[k]).T)),H[k])).T )
        
    #Scale V
    alpha = np.sqrt(PT/np.sum([np.trace(np.dot(V[i],V[i].conjugate().T)) for i in range(K)]))
    for k in range(K):  #Scale V_k
        V[k] = alpha * V[k]           
    
    iteration = []
    sum_rate = []
    t = 0
    #while (err > epsilon) or t > Imax:
    while  t < 100:
        U = update_Uk(A, U, V, H, K, Nr)
        W, W_old = update_Wk(U, V, W_old, W, K, d)
        V = update_Vk(V, W, U, H, K, Nt, sigma)
        #err = abs(np.sum(np.fromiter([np.log(np.linalg.det(W[i])) for i in range(K)],complex)) -  np.sum(np.fromiter([np.log(np.linalg.det(W_old[i])) for i in range(K)],complex)))
        sum_rate.append(np.sum(np.fromiter([np.log(np.linalg.det(W[k])) for k in range(K)],complex)))
        t += 1;
        iteration.append(t) 
        
    rate.append(sum_rate[t-1])

#    plt.plot(iteration,sum_rate)
#    plt.xlabel("iteration")
#    plt.ylabel("rate")
#    plt.show()  
    
print(np.array(rate).mean())
      
    
    
    
    