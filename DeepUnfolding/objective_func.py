import numpy as np
import UW_gradient as UW_gradi


########################### The whole objective function
def objective_func_whole(delta_noise, X_U, X_W, X_V, Y_U, Y_W, Y_V, Z_U, Z_W, Z_V, O_U, O_V, H, Nr, Nt, d, Pt, sigma, User, Layer, V_initial):
    obj=0
    
############ Forward 
    A=[]
    for l in range(Layer):
        A.append([])
        for k in range(User):
            A[l].append( np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) )
        
    E=[]
    for l in range(Layer):
        E.append([])
        for k in range(User):
            E[l].append( np.matrix(np.zeros((d,d),dtype=np.complex)) )
    
    B=[]
    for l in range(Layer):
        B.append( np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) )
        
    I_A=[]
    for l in range(Layer):
        I_A.append([])
        for k in range(User):
            I_A[l].append( np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) )
        
    I_E=[]
    for l in range(Layer):
        I_E.append([])
        for k in range(User):
            I_E[l].append( np.matrix(np.zeros((d,d),dtype=np.complex)) )
    
    I_B=[]
    for l in range(Layer):
        I_B.append( np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) )
        
    U=[]
    for l in range(Layer):
        U.append([])
        for k in range(User):
            U[l].append( np.matrix(np.zeros((Nr,d),dtype=np.complex)) )
        
    W=[]
    for l in range(Layer):
        W.append([])
        for k in range(User):
            W[l].append( np.matrix( np.zeros((d,d))+np.zeros((d,d))*1j,dtype=np.complex ) )    
    
    V=[]
    for l in range(Layer):
        V.append([])
        for k in range(User):
            V[l].append( np.matrix(np.zeros((Nt,d),dtype=np.complex)) ) 
    
    for k in range(User):
        V[0][k]=V_initial[k]
        
    V1=[]
    for l in range(Layer):
        V1.append([])
        for k in range(User):
            V1[l].append( np.matrix( np.zeros((Nt,d))+np.zeros((Nt,d))*1j,dtype=np.complex ) )
            
    for k in range(User):
        V1[0][k]=V[0][k]#/np.linalg.norm(V[0][k])
    
########## Forward: Update till 'Layer-2'    
    for l in range(1,Layer-1):                
        for k in range(User):
            for m in range(User):
                A[l-1][k] = A[l-1][k] + sigma**2/Pt*np.trace(V1[l-1][m]*V1[l-1][m].H)*np.matrix(np.identity((Nr),dtype=np.complex)) \
                + H[k]*V1[l-1][m]*V1[l-1][m].H*H[k].H
            
            A[l-1][k] = A[l-1][k] + delta_noise*np.matrix(np.ones((Nr,Nr),dtype=np.complex))
            I_A[l-1][k] = np.mat(np.diag(np.squeeze(1/np.diagonal(A[l-1][k]))))
            
        for k in range(User):
            U[l][k]=(I_A[l-1][k]*X_U[l][k] + A[l-1][k]*Y_U[l][k] + Z_U[l][k])*H[k]*V1[l-1][k]+O_U[l][k]
            
        for k in range(User):
            E[l][k] = np.matrix(np.identity((d),dtype=np.complex)) - U[l][k].H*H[k]*V1[l-1][k]
            E[l][k] = E[l][k] + delta_noise*np.matrix(np.ones((d,d),dtype=np.complex))
            I_E[l][k] = np.mat(np.diag(np.squeeze(1/np.diagonal(E[l][k]))))
            
        for k in range(User):
            W[l][k] = I_E[l][k]*X_W[l][k]+E[l][k]*Y_W[l][k]+Z_W[l][k]
        
        for k in range(User):
            B[l] = B[l] + sigma**2/Pt*np.trace(U[l][k]*W[l][k]*U[l][k].H)*np.matrix(np.identity((Nt),dtype=np.complex)) \
            + H[k].H*U[l][k]*W[l][k]*U[l][k].H*H[k]
        
        B[l] = B[l]+delta_noise*np.matrix(np.ones((Nt,Nt),dtype=np.complex))
        I_B[l] = np.mat(np.diag(np.squeeze(1/np.diagonal(B[l]))))
        
        for k in range(User):
            V[l][k] = (I_B[l]*X_V[l][k]+B[l]*Y_V[l][k]+Z_V[l][k])*H[k].H*U[l][k]*W[l][k]+O_V[l][k]
        
        for k in range(User):
            V1[l][k] = V[l][k] #*np.sqrt(Pt)/np.sqrt(temp_trace)      
            

########## Forward: update U[Layer-1]    
    temp_func4=np.matrix(np.zeros((Nt,Nt),dtype=np.complex))
    for k in range(User):
        temp_func4 = temp_func4 + V1[Layer-2][k]*V1[Layer-2][k].H 
    
    for k in range(User):
        A[Layer-2][k] = (sigma**2)/Pt*np.trace(temp_func4)*np.matrix(np.identity((Nr),dtype=np.complex)) + H[k]*temp_func4*H[k].H
        A[Layer-2][k] = A[Layer-2][k] + delta_noise*np.matrix(np.ones((Nr,Nr),dtype=np.complex))
        I_A[Layer-2][k] = np.mat(np.diag(np.squeeze(1/np.diagonal(A[Layer-2][k]))))
        
    
    for k in range(User):
        U[Layer-1][k] = ( I_A[Layer-2][k]*X_U[Layer-1][k]+A[Layer-2][k]*Y_U[Layer-1][k]+Z_U[Layer-1][k] )*H[k]*V1[Layer-2][k]+O_U[Layer-1][k]

########## Forward: update W[Layer-1]             
    for k in range(User):
        E[Layer-1][k] = np.matrix(np.identity((d),dtype=np.complex))-U[Layer-1][k].H*H[k]*V1[Layer-2][k]  
        E[Layer-1][k] = E[Layer-1][k] + delta_noise*np.matrix(np.ones((d,d),dtype=np.complex))
        I_E[Layer-1][k] = np.mat(np.diag(np.squeeze(1/np.diagonal(E[Layer-1][k]))))
    
    for k in range(User):
        W[Layer-1][k] = I_E[Layer-1][k]*X_W[Layer-1][k]+E[Layer-1][k]*Y_W[Layer-1][k]+Z_W[Layer-1][k]  

########## Forward: update V[Layer-1]     
    temp_func3=np.matrix(np.zeros((Nt,Nt),dtype=np.complex))
    for j in range(User):
        temp_func3 += (sigma**2)/Pt*np.trace(U[Layer-1][j]*W[Layer-1][j]*U[Layer-1][j].H)*np.matrix(np.identity((Nt),dtype=np.complex))  \
        + H[j].H*U[Layer-1][j]*W[Layer-1][j]*U[Layer-1][j].H*H[j]

    C=[0 for i in range(User)]    
    for k in range(User):
        C[k]=temp_func3  

    for k in range(User):
        V[Layer-1][k] = C[k].I*H[k].H*U[Layer-1][k]*W[Layer-1][k] 

########## The last layer of the objective function    
    temp2=0    
    for j in range(User):
        temp2 += np.trace(V[Layer-1][j]*V[Layer-1][j].H)
    
    for k in range(User):        ########## To make the constraint satisfied
        V[Layer-1][k] = V[Layer-1][k]*np.sqrt(Pt)/np.sqrt(temp2)

    temp1=[]
    for k in range(User):
        temp1.append(np.matrix( np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) ))

    D=[]
    for l in range(User):
        D.append( np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) )
        
    for k in range(User):
        for i in range(User):
            if (i != k):
                temp1[k] += H[k]*V[Layer-1][i]*V[Layer-1][i].H*H[k].H
        D[k] = temp1[k]+sigma**2*np.matrix(np.identity((Nr),dtype=np.complex))
    
    for k in range(User):
        obj += np.log( np.linalg.det( np.matrix(np.identity((Nr),dtype=np.complex))+H[k]*V[Layer-1][k]*V[Layer-1][k].H*H[k].H*D[k].I  ) ) 
    return obj

########################### Objective function for the laster Layer of U and W
def objective_func_UW(Nr,Nt,d,Pt,sigma,User,H,W,U):
    obj=0
    
    temp_func3=np.matrix(np.zeros((Nt,Nt),dtype=np.complex))
    for j in range(User):
        temp_func3 += (sigma**2)/Pt*np.trace(U[j]*W[j]*U[j].H)*np.identity(Nt)+H[j].H*U[j]*W[j]*U[j].H*H[j]

    C=[0 for i in range(User)]    
    for k in range(User):
        C[k]=temp_func3  
    
    V=[]
    for k in range(User):
        V.append(np.matrix(np.zeros((Nt,d),dtype=np.complex)))

    for k in range(User):
        V[k] = C[k].I*H[k].H*U[k]*W[k] 
    
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
    
    for k in range(User):
        obj += np.log( np.linalg.det( np.identity(Nr)+H[k]*V[k]*V[k].H*H[k].H*B[k].I  ) ) 
    return obj