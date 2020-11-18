import tensorflow as tf
import numpy as np
import objective_func as obj_func

######################## Calculate the gradient of U in the last layer 
def U_conj_gradient_lastlayer(Nr,Nt,d,Pt,sigma,User,H,W,U):
    
    
    temp3=np.matrix(np.zeros((Nt,Nt),dtype=np.complex))
    for j in range(User):
        temp3 += (sigma**2)/Pt*np.trace(U[j]*W[j]*U[j].H)*np.identity(Nt) \
        +H[j].H*U[j]*W[j]*U[j].H*H[j]
    
    C=[0 for i in range(User)]    
    for k in range(User):
        C[k]=temp3  
    
    
    V=[]
    for k in range(User):
        V.append(np.matrix(np.zeros((Nt,d),dtype=np.complex)))
    
    for k in range(User):
        V[k] = C[k].I*H[k].H*U[k]*W[k]
    
    T=[0 for i in range(User)]
    for k in range(User):
        T[k] = H[k].H*U[k]*W[k]
    
    temp2=0    
    for j in range(User):
        temp2 += np.trace(V[j]*V[j].H)
    
    temp1=[]
    for k in range(User):
        temp1.append(np.matrix( np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) ))
    
    B=[0 for i in range(User)]    
    for k in range(User):
        for i in range(User):
            if (i != k):
                temp1[k] += H[k]*V[i]*V[i].H*H[k].H
        B[k] = temp1[k]+sigma**2*temp2/Pt*np.identity(Nr)
    
    
    X=[0 for i in range(User)]    
    for k in range(User):
        X[k] = np.identity(Nr)+H[k]*V[k]*V[k].H*H[k].H*B[k].I
    
    D=[0 for i in range(User)]    
    for k in range(User):
        D[k] = H[k].H*B[k].I*X[k].I*H[k]*V[k]*V[k].H*H[k].H*B[k].I*H[k]
    
    E=[0 for i in range(User)]    
    for k in range(User):
        E[k] = B[k].I*X[k].I*H[k]*V[k]*V[k].H*H[k].H*B[k].I
    
    #The Gradient of U, which contains 15 items     
    #The First five item
    First_item=[]
    for k in range(User):
        First_item.append(np.matrix(np.zeros((Nr,d),dtype=np.complex)))
        
    for m in range(User):
        for k in range(User):
            First_item[m] += -sigma**2/Pt*np.trace( C[k].H.I*H[k].H*B[k].I*X[k].I*H[k]*V[k]*T[k].H*C[k].H.I ) *U[m]*W[m].H \
                             -sigma**2/Pt*np.trace( C[k].I*T[k]*V[k].H*H[k].H*B[k].I*X[k].I*H[k]*C[k].I ) *U[m]*W[m]  \
                             -H[m]*C[k].H.I*H[k].H*B[k].I*X[k].I*H[k]*V[k]*T[k].H*C[k].H.I*H[m].H*U[m]*W[m].H \
                             -H[m]*C[k].I*T[k]*V[k].H*H[k].H*B[k].I*X[k].I*H[k]*C[k].I*H[m].H*U[m]*W[m] 
                             
        First_item[m] += H[m]*C[m].H.I*H[m].H*B[m].I*X[m].I*H[m]*V[m]*W[m].H
    
    #The Second five item                         
    Second_item=[]
    for k in range(User):
        Second_item.append(np.matrix(np.zeros((Nr,d),dtype=np.complex)))
        
    for l in range(User):
        for k in range(User):
            for m in range(User):
                if (m != k):
                    Second_item[l] += -sigma**2/Pt*np.trace( D[k]*V[m]*T[m].H*C[m].H.I*C[m].H.I )*U[l]*W[l].H \
                                      -sigma**2/Pt*np.trace( D[k]*C[m].I*C[m].I*T[m]*V[m].H )*U[l]*W[l] \
                                      -H[l]*C[m].H.I*D[k]*V[m]*T[m].H*C[m].H.I*H[l].H*U[l]*W[l].H \
                                      -H[l]*C[m].I*T[m]*V[m].H*D[k]*C[m].I*H[l].H*U[l]*W[l]
    
    for m in range(User):
        for k in range(User):
            if (m != k):    
                 Second_item[m] += H[m]*C[m].H.I*D[k]*V[m]*W[m].H
     
    
    #The Third five item                         
    Third_item=[]
    for k in range(User):
        Third_item.append(np.matrix(np.zeros((Nr,d),dtype=np.complex)))
        
    for l in range(User):
        for k in range(User):
            for m in range(User):
                Third_item[l] +=  -sigma**2/Pt*np.trace( E[k] )*np.trace( V[m]*T[m].H*C[m].H.I*C[m].H.I )*sigma**2/Pt*U[l]*W[l].H \
                                  -sigma**2/Pt*np.trace( E[k] )*sigma**2/Pt*np.trace( C[m].I*C[m].I*T[m]*V[m].H )*U[l]*W[l] \
                                  -sigma**2/Pt*np.trace( E[k] )*H[l]*C[m].H.I*V[m]*T[m].H*C[m].H.I*H[l].H*U[l]*W[l].H \
                                  -sigma**2/Pt*np.trace( E[k] )*H[l]*C[m].I*T[m]*V[m].H*C[m].I*H[l].H*U[l]*W[l]
    
            Third_item[l] += sigma**2/Pt*np.trace( E[k] )*H[l]*C[l].I.H*V[l]*W[l].H
                   
    #Gradient of U             
    Gradient_U=[0 for i in range(User)]
    for k in range(User):
        Gradient_U[k] = First_item[k]-Second_item[k]-Third_item[k] 
        
    return Gradient_U       



######################## Calculate the gradient of W in the last layer 
def W_conj_gradient_lastlayer(Nr,Nt,d,Pt,sigma,User,H,W,U):  
    
    C = np.mat(np.zeros((Nt, Nt), dtype=np.complex))
    for k in range(User):
        C = C + (sigma**2)/Pt*np.trace(U[k] * W[k] * U[k].H)*np.eye(Nt) + H[k].H * U[k] * W[k] * U[k].H * H[k]
        
    V=[]
    for k in range(User):
        V.append(np.matrix(np.zeros((Nt,d),dtype=np.complex)))
    
    for k in range(User):
        V[k] = C.I*H[k].H*U[k]*W[k]
    
    
    B = [0 for i in range(User)]
    X = [0 for i in range(User)]
    D = [0 for i in range(User)]
    E = [0 for i in range(User)]
    T = [0 for i in range(User)]
    for k in range(User):
        B[k] = np.mat(np.zeros((Nr, Nr)))
        for m in range(User):
            B[k] = B[k] + sigma*sigma/Pt*np.trace(V[m] * V[m].H)*np.eye(Nr)
            B[k] = B[k] + H[k] * V[m] * V[m].H * H[k].H
        B[k] = B[k] - H[k] * V[k] * V[k].H * H[k].H
    
    for k in range(User):
        X[k] = np.eye(Nr) + H[k] * V[k] * V[k].H * H[k].H * B[k].I
        E[k] = B[k].I * X[k].I * H[k] * V[k] * V[k].H * H[k].H * B[k].I
        D[k] = H[k].H * E[k] * H[k]
        T[k] = H[k].H * U[k] * W[k]
    
    tempobj=0
    for k in range(User):
        tempobj+=np.log( np.linalg.det(X[k]))
    
    tempE = 0
    for k in range(User):
        tempE = tempE + np.trace(E[k])
        
    Gradient_W = [0 for i in range(User)]
    for l in range(User):
        Gradient_W[l] = U[l].H * H[l] * C.I.H * H[l].H * B[l].I * X[l].I * H[l] * V[l]
        for k in range(User):
            Gradient_W[l] = Gradient_W[l] - np.trace(X[k].I * H[k] * V[k] * T[k].H * C.I.H * C.I.H * H[k].H * B[k].I ) * sigma*sigma/Pt * U[l].H * U[l] \
                         - U[l].H * H[l] * C.I.H * H[k].H * B[k].I * X[k].I * H[k] * V[k] * T[k].H * C.I.H * H[l].H * U[l] \
                         - U[l].H * H[l] * C.I.H * H[k].H * E[k] * H[k] * V[l]
            for m in range(User):
                Gradient_W[l] = Gradient_W[l] + np.trace( E[k] * H[k] * V[m] * T[m].H * C.I.H * C.I.H * H[k].H ) * sigma*sigma/Pt * U[l].H * U[l]   \
                + U[l].H * H[l] * C.I.H * H[k].H * E[k] * H[k] * V[m] * T[m].H * C.I.H * H[l].H * U[l]            
            Gradient_W[l] = Gradient_W[l] - np.trace( E[k] * H[k] * V[k] * T[k].H * C.I.H * C.I.H * H[k].H ) * sigma*sigma/Pt * U[l].H * U[l]   \
            - U[l].H * H[l] * C.I.H * H[k].H * E[k] * H[k] * V[k] * T[k].H * C.I.H * H[l].H * U[l]
        
        Gradient_W[l] = Gradient_W[l] - sigma*sigma/Pt*tempE* U[l].H * H[l] * C.I.H * V[l]
        for n in range(User):
            Gradient_W[l] = Gradient_W[l] + sigma*sigma/Pt*tempE * ( sigma*sigma/Pt * np.trace(V[n] * T[n].H * C.I.H * C.I.H )* U[l].H * U[l] \
            + U[l].H * H[l] * C.I.H * V[n] * T[n].H  * C.I.H * H[l].H * U[l] )
        
        
        Gradient_W[l] = Gradient_W[l] + U[l].H * H[l] * C.I.H * H[l].H * E[l] * H[l] * V[l] 
             
    return Gradient_W 
 
    