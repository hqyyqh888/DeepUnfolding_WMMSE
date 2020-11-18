import numpy.matlib
import numpy as np
import objective_func as obj_func
import UW_gradient as UW_gradi
import UW_conj_gradient as UW_conj_gradi 
import pickle 
import datetime
import matplotlib.pyplot as plt
import generate_channel

# %%
########################### Forward propagation, here the function 'forward' till 'Layer-2', the layer are indexed as 0,1,2..., Layer-2, Layer-1.
def forward():
    global User, Layer, Nt, Nr, d, Pt, sigma, delta_noise
    global X_U, X_W, X_V, Y_U, Y_W, Y_V, Z_U, Z_W, Z_V, O_U, O_V, H, U, W, V, A, E, B, V1, I_A, I_E, I_B


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
     
    return


# %%
def back_propagation():
    global User, Layer, Nt, Nr, d, Pt, sigma, delta_noise
    global X_U, X_W, X_V, Y_U, Y_W, Y_V, Z_U, Z_W, Z_V, O_U, O_V, H, U, W, V, A, E, B, V1
    global G_UH_X, G_UH_Y, G_UH_Z, G_UH_O, G_WH_X, G_WH_Y, G_WH_Z, G_VH_X, G_VH_Y, G_VH_Z, G_VH_O
############################ Initialize G J L M    
    G_UH=[ [ np.matrix(np.zeros((Nr,d),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
    G_WH=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_VH=[ [ np.matrix(np.zeros((Nt,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_VH_temp=[ [ np.matrix(np.zeros((Nt,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    G_U_intermediate= [ np.matrix(np.zeros((Nr,d),dtype=np.complex))  for i in range(User) ] 
    G_W_intermediate= [ np.matrix(np.zeros((d,d),dtype=np.complex))  for i in range(User) ] 
    
    G_U=[ [ np.matrix(np.zeros((d,Nr),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
    G_W=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_V=[ [ np.matrix(np.zeros((d,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_V_temp=[ [ np.matrix(np.zeros((d,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    
    J_UH=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    J_WH=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    J_VH=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    J_U=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    J_W=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    J_V=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    L_UH=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    L_VH=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    L_U=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    L_V=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    M_UH=[ [ np.matrix(np.zeros((Nt,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    M_VH=[ [ np.matrix(np.zeros((Nr,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    M_U=[ [ np.matrix(np.zeros((d,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    M_V=[ [ np.matrix(np.zeros((Nt,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    
    
############################## The Gradient of the last layer
    G_UH[Layer-1] = UW_gradi.U_gradient_lastlayer(Nr,Nt,d,Pt,sigma,User,H,W[Layer-1],U[Layer-1])
    G_WH[Layer-1] = UW_gradi.W_gradient_lastlayer(Nr,Nt,d,Pt,sigma,User,H,W[Layer-1],U[Layer-1])

    for k in range(User):
        G_U[Layer-1][k] = G_UH[Layer-1][k].H
        G_W[Layer-1][k] = G_WH[Layer-1][k].H
    
    for n in range(User):
        J_W[Layer-1][n] = np.multiply( ( X_W[Layer-1][n]*G_W[Layer-1][n] ) , np.multiply( -I_E[Layer-1][n] , I_E[Layer-1][n] ).T ) 
        J_WH[Layer-1][n] = np.multiply( ( G_WH[Layer-1][n]*X_W[Layer-1][n].H ) , np.multiply( -I_E[Layer-1][n].H , I_E[Layer-1][n].H ).T ) 
    
    for n in range(User):
        G_U[Layer-1][n] =  G_U[Layer-1][n] - J_WH[Layer-1][n]*V1[Layer-2][n].H*H[n].H - G_WH[Layer-1][n]*Y_W[Layer-1][n].H*V1[Layer-2][n].H*H[n].H 
        G_UH[Layer-1][n] = G_UH[Layer-1][n] - H[n]*V1[Layer-2][n]*J_W[Layer-1][n] - H[n]*V1[Layer-2][n]*Y_W[Layer-1][n]*G_W[Layer-1][n] 
    
    
####### Update J_U  L_U  M_U  J_W  in the last layer  
    for n in range(User):
        J_U[Layer-1][n] = np.multiply( ( X_U[Layer-1][n]*H[n]*V1[Layer-2][n]*G_U[Layer-1][n] ) , np.multiply( -I_A[Layer-2][n] , I_A[Layer-2][n] ).T )
        J_UH[Layer-1][n] = np.multiply( ( G_UH[Layer-1][n]*V1[Layer-2][n].H*H[n].H*X_U[Layer-1][n].H ) , np.multiply( -I_A[Layer-2][n].H , I_A[Layer-2][n].H ).T )
        
        L_U[Layer-1][n] = Y_U[Layer-1][n]*H[n]*V1[Layer-2][n]*G_U[Layer-1][n]
        L_UH[Layer-1][n] = G_UH[Layer-1][n]*V1[Layer-2][n].H*H[n].H*Y_U[Layer-1][n].H
    
        M_U[Layer-1][n] = G_U[Layer-1][n]*( (I_A[Layer-2][n])*X_U[Layer-1][n] + A[Layer-2][n]*Y_U[Layer-1][n] + Z_U[Layer-1][n]  )*H[n]
        M_UH[Layer-1][n] = H[n].H*( (I_A[Layer-2][n])*X_U[Layer-1][n] + A[Layer-2][n]*Y_U[Layer-1][n] + Z_U[Layer-1][n]  ).H*G_UH[Layer-1][n]
         
############################### The whole iteration here, i.e., the recursion, interlace,  l = Layer-2, Layer-3 ,..., 1
    Layer_inverse=[]
    for l in range(1,Layer-1):
        Layer_inverse.append(l)
    
    Layer_inverse.reverse()
    for l in Layer_inverse:   
############# The whole power        
        temp_trace=0
        for k in range(User):
            temp_trace = temp_trace + np.trace( V[l][k]*V[l][k].H )
############# Update G_V 
####### Update G_V_temp
        for n in range(User):         
            for k in range(User):            
                G_V_temp[l][n] = G_V_temp[l][n] + np.trace( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*sigma*sigma/Pt*V1[l][n].H  \
                + V1[l][n].H*H[k].H*( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*H[k]
                
                G_VH_temp[l][n] = G_VH_temp[l][n] + np.trace( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*sigma*sigma/Pt*V1[l][n]  \
                + H[k].H*( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*H[k]*V1[l][n]
                
            G_V_temp[l][n] =  G_V_temp[l][n] - J_W[l+1][n]*U[l+1][n].H*H[n] - Y_W[l+1][n]*G_W[l+1][n]*U[l+1][n].H*H[n] + M_U[l+1][n] 
            G_VH_temp[l][n] = G_VH_temp[l][n] - H[n].H*U[l+1][n]*J_WH[l+1][n] - H[n].H*U[l+1][n]*G_WH[l+1][n]*Y_W[l+1][n].H + M_UH[l+1][n]  
    
####### Update G_V normalized by the power of all the users
        for n in range(User):         
            G_V[l][n] = G_V_temp[l][n]           
            G_VH[l][n] = G_VH_temp[l][n]

    
####### Update J_V  L_V  M_V 
        for n in range(User):
            J_V[l][n] = np.multiply(  ( X_V[l][n]*H[n].H*U[l][n]*W[l][n]*G_V[l][n] ) , np.multiply( -I_B[l] , I_B[l] ).T )
            L_V[l][n] = Y_V[l][n]*H[n].H*U[l][n]*W[l][n]*G_V[l][n]
            M_V[l][n] = ( (I_B[l])*X_V[l][n] + B[l]*Y_V[l][n] + Z_V[l][n]  )*H[n].H
            
            J_VH[l][n] = np.multiply(  ( G_VH[l][n]*W[l][n].H*U[l][n].H*H[n]*X_V[l][n].H ) , np.multiply( -I_B[l].H , I_B[l].H ).T )
            L_VH[l][n] = G_VH[l][n]*W[l][n].H*U[l][n].H*H[n]*Y_V[l][n].H
            M_VH[l][n] = H[n]*( (I_B[l])*X_V[l][n] + B[l]*Y_V[l][n] + Z_V[l][n]  ).H
    
####### Update G_W        
        for n in range(User):
            for k in range(User): 
                G_W[l][n] = G_W[l][n] + np.trace( J_V[l][k]+L_V[l][k] )*sigma*sigma/Pt*U[l][n].H*U[l][n]   \
                + U[l][n].H*H[n]*( J_V[l][k]+L_V[l][k] )*H[n].H*U[l][n]
    
                G_WH[l][n] = G_WH[l][n] + np.trace( J_VH[l][k]+L_VH[l][k] )*sigma*sigma/Pt*U[l][n].H*U[l][n] \
                + U[l][n].H*H[n]*( J_VH[l][k]+L_VH[l][k] )*H[n].H*U[l][n]
                
            G_W[l][n] =  G_W[l][n] + G_V[l][n]*M_V[l][n]*U[l][n] 
            G_WH[l][n] = G_WH[l][n] + U[l][n].H*M_V[l][n].H*G_VH[l][n]
    
####### Update J_W     
        for n in range(User):
            J_W[l][n] = np.multiply( X_W[l][n]*G_W[l][n] , np.multiply( -I_E[l][n] , I_E[l][n] ).T )
            J_WH[l][n] = np.multiply( G_WH[l][n]*X_W[l][n].H, np.multiply( -I_E[l][n].H , I_E[l][n].H ).T )
    
####### Update G_U     
        for n in range(User):
            for k in range(User):            
                G_U[l][n] = G_U[l][n] + np.trace( J_V[l][k]+L_V[l][k] )*sigma*sigma/Pt*W[l][n]*U[l][n].H     \
                + np.trace( J_VH[l][k]+L_VH[l][k] )*sigma*sigma/Pt*W[l][n].H*U[l][n].H                     \
                + W[l][n]*U[l][n].H*H[n]*( J_V[l][k] + L_V[l][k] )*H[n].H + W[l][n].H*U[l][n].H*H[n]*(J_VH[l][k] + L_VH[l][k])*H[n].H  
    
                G_UH[l][n] = G_UH[l][n] + np.trace( J_V[l][k]+L_V[l][k] )*sigma*sigma/Pt*U[l][n]*W[l][n]     \
                + np.trace( J_VH[l][k]+L_VH[l][k] )*sigma*sigma/Pt*U[l][n]*W[l][n].H                     \
                + H[n]*( J_V[l][k] + L_V[l][k] )*H[n].H*U[l][n]*W[l][n] + H[n]*( J_VH[l][k] + L_VH[l][k] )*H[n].H*U[l][n]*W[l][n].H                
    
            G_U[l][n] =  G_U[l][n] - J_WH[l][n]*V1[l-1][n].H*H[n].H - G_WH[l][n]*Y_W[l][n].H*V1[l-1][n].H*H[n].H + W[l][n]*G_V[l][n]*M_V[l][n] 
            G_UH[l][n] = G_UH[l][n] - H[n]*V1[l-1][n]*J_W[l][n] - H[n]*V1[l-1][n]*Y_W[l][n]*G_W[l][n] + M_VH[l][n]*G_VH[l][n]*W[l][n].H
        
####### Update J_U  L_U  M_U     
        for n in range(User):
            J_U[l][n] = np.multiply( X_U[l][n]*H[n]*V1[l-1][n]*G_U[l][n] , np.multiply( -I_A[l-1][n] , I_A[l-1][n] ).T )
            L_U[l][n] = Y_U[l][n]*H[n]*V1[l-1][n]*G_U[l][n]
            M_U[l][n] = G_U[l][n]*( (I_A[l-1][n])*X_U[l][n] + A[l-1][n]*Y_U[l][n] + Z_U[l][n]  )*H[n]
            
            J_UH[l][n] = np.multiply( G_UH[l][n]*V1[l-1][n].H*H[n].H*X_U[l][n].H , np.multiply( -I_A[l-1][n].H , I_A[l-1][n].H ).T )
            L_UH[l][n] = G_UH[l][n]*V1[l-1][n].H*H[n].H*Y_U[l][n].H
            M_UH[l][n] = H[n].H*( (I_A[l-1][n])*X_U[l][n] + A[l-1][n]*Y_U[l][n] + Z_U[l][n]  ).H*G_UH[l][n]
            
    
############################### The first layer that only contains V[0],i.e., l=0  
    for l in range(1): 
        
        temp_trace=0
        for k in range(User):
            temp_trace = temp_trace + np.trace( V[l][k]*V[l][k].H )
        
        for n in range(User):         
            for k in range(User):            
                G_V_temp[l][n] = G_V_temp[l][n] + np.trace( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*sigma*sigma/Pt*V1[l][n].H  \
                + V1[l][n].H*H[k].H*( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*H[k]
                
                G_VH_temp[l][n] = G_VH_temp[l][n] + np.trace( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*sigma*sigma/Pt*V1[l][n]  \
                + H[k].H*( J_U[l+1][k]+L_U[l+1][k]+J_UH[l+1][k]+L_UH[l+1][k] )*H[k]*V1[l][n]
                
            G_V_temp[l][n] =  G_V_temp[l][n] - J_W[l+1][n]*U[l+1][n].H*H[n] - Y_W[l+1][n]*G_W[l+1][n]*U[l+1][n].H*H[n] + M_U[l+1][n] 
            G_VH_temp[l][n] = G_VH_temp[l][n] - H[n].H*U[l+1][n]*J_WH[l+1][n] - H[n].H*U[l+1][n]*G_WH[l+1][n]*Y_W[l+1][n].H + M_UH[l+1][n]  
    
####### Update G_V normalized by the power of all the users
        for n in range(User):         
            G_V[l][n] = G_V_temp[l][n]           
            G_VH[l][n] = G_VH_temp[l][n]
    
############################### The gradients of trainable parameters X Y Z O    
####################  Layer-2 --> 1
    Layer_inverse_part=[]
    for l in range(0,Layer-2):
        Layer_inverse_part.append(l)
    
    Layer_inverse_part.reverse()
    
    for l in Layer_inverse_part:       
        for k in range(User):
            G_VH_X[l+1][k] = I_B[l+1].H*G_VH[l+1][k]*W[l+1][k].H*U[l+1][k].H*H[k]
#            G_VH_Y[l+1][k] = B[l+1].H*G_VH[l+1][k]*W[l+1][k].H*U[l+1][k].H*H[k]
            G_VH_Z[l+1][k] = G_VH[l+1][k]*W[l+1][k].H*U[l+1][k].H*H[k]
            G_VH_O[l+1][k] = G_VH[l+1][k]
            
            G_WH_X[l+1][k] = I_E[l+1][k].H*G_WH[l+1][k]
#            G_WH_Y[l+1][k] = E[l+1][k].H*G_WH[l+1][k]
            G_WH_Z[l+1][k] = G_WH[l+1][k]
             
            G_UH_X[l+1][k] = I_A[l][k].H*G_UH[l+1][k]*V[l][k].H*H[k].H
#            G_UH_Y[l+1][k] = A[l][k].H*G_UH[l+1][k]*V[l][k].H*H[k].H
            G_UH_Z[l+1][k] = G_UH[l+1][k]*V[l][k].H*H[k].H
            G_UH_O[l+1][k] = G_UH[l+1][k]
            
####################  Layer-1 
    for k in range(User):
        G_WH_X[Layer-1][k] = I_E[Layer-1][k].H*G_WH[Layer-1][k]
#        G_WH_Y[Layer-1][k] = E[Layer-1][k].H*G_WH[Layer-1][k]
        G_WH_Z[Layer-1][k] = G_WH[Layer-1][k]
         
        G_UH_X[Layer-1][k] = I_A[Layer-2][k].H*G_UH[Layer-1][k]*V[Layer-2][k].H*H[k].H
#        G_UH_Y[Layer-1][k] = A[Layer-2][k].H*G_UH[Layer-1][k]*V[Layer-2][k].H*H[k].H
        G_UH_Z[Layer-1][k] = G_UH[Layer-1][k]*V[Layer-2][k].H*H[k].H
        G_UH_O[Layer-1][k] = G_UH[Layer-1][k]

    return

 
# %%
######################### System configuration
global User, Layer, Nt, Nr, d, Pt, sigma, snr_dB, delta_noise
global X_U, X_W, X_V, Y_U, Y_W, Y_V, Z_U, Z_W, Z_V, O_U, O_V, H, U, W, V, A, B, E, I_A, I_E, I_B, V1 
global G_UH_X, G_UH_Y, G_UH_Z, G_UH_O, G_WH_X, G_WH_Y, G_WH_Z, G_VH_X, G_VH_Y, G_VH_Z, G_VH_O

User = 30  #The number of users
Layer = 5 #The number of layers
Nt = 64 #The number of transmit antennas
Nr = 2 #The number of receive antennas
d = 2  #The number of data stream

snr_dB = 20 #SNR in dB
Pt = 100
sigma = 1
batch = 10
number_of_sample = 10**(7)

delta_noise = 0
scale_factor_X = 0.1
scale_factor_Y = 0
scale_factor_Z = 0.1
scale_factor_O = 0.1
#step_size = 1

count = 0
obj_value_temp = 0
obj_value = np.zeros(10**5)+np.zeros(10**5)*1j

Layer_inverse1=[]
for l in range(0,Layer-2):
    Layer_inverse1.append(l)

Layer_inverse1.reverse()

starttime = datetime.datetime.now()

# %%
############################ Initialize trainable parameters X Y Z O and their gradients

X_U=[ [ scale_factor_X*np.matrix( -1-1j + 2*np.random.random((Nr,Nr)) + 2*np.random.random((Nr,Nr))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]
X_W=[ [ scale_factor_X*np.matrix( -1-1j + 2*np.random.random((d,d)) + 2*np.random.random((d,d))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]
X_V=[ [ scale_factor_X*np.matrix( -1-1j + 2*np.random.random((Nt,Nt)) + 2*np.random.random((Nt,Nt))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]

Y_U=[ [ scale_factor_Y*np.matrix( -1-1j + 2*np.random.random((Nr,Nr)) + 2*np.random.random((Nr,Nr))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]
Y_W=[ [ scale_factor_Y*np.matrix( -1-1j + 2*np.random.random((d,d)) + 2*np.random.random((d,d))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]
Y_V=[ [ scale_factor_Y*np.matrix( -1-1j + 2*np.random.random((Nt,Nt)) + 2*np.random.random((Nt,Nt))*1j,dtype=np.complex )  for i in range(User) ] for j in range(Layer) ]

Z_U=[ [ scale_factor_Z*np.matrix( -1-1j + 2*np.random.random((Nr,Nr)) + 2*np.random.random((Nr,Nr))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]
Z_W=[ [ scale_factor_Z*np.matrix( -1-1j + 2*np.random.random((d,d)) + 2*np.random.random((d,d))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]
Z_V=[ [ scale_factor_Z*np.matrix( -1-1j + 2*np.random.random((Nt,Nt)) + 2*np.random.random((Nt,Nt))*1j,dtype=np.complex )  for i in range(User) ] for j in range(Layer) ]

O_U=[ [ scale_factor_O*np.matrix( -1-1j + 2*np.random.random((Nr,d)) + 2*np.random.random((Nr,d))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]
O_V=[ [ scale_factor_O*np.matrix( -1-1j + 2*np.random.random((Nt,d)) + 2*np.random.random((Nt,d))*1j,dtype=np.complex ) for i in range(User) ] for j in range(Layer) ]


G_UH_X_batch=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
G_UH_Y_batch=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
G_UH_Z_batch=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
G_UH_O_batch=[ [ np.matrix(np.zeros((Nr,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]

G_WH_X_batch=[ [ np.matrix(np.zeros((d,d),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
G_WH_Y_batch=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
G_WH_Z_batch=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]

G_VH_X_batch=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
G_VH_Y_batch=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
G_VH_Z_batch=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
G_VH_O_batch=[ [ np.matrix(np.zeros((Nt,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]

# %%
#fr = open('../data_channel/data_channel_T8_R2.pkl','rb') 
for i in range(10**(4)):     
#    H = pickle.load(fr)   
#    print('H', i, H)   
   
    H=[]
    for k in range(User):
        H.append( np.matrix( np.random.randn(Nr,Nt) + np.random.randn(Nr,Nt)*1j,dtype=np.complex ) )
    
    U=[]
    for l in range(Layer):
        U.append([])
        for k in range(User):
            U[l].append( np.matrix( np.zeros((Nr,d))+np.zeros((Nr,d))*1j,dtype=np.complex ) )
        
    W=[]
    for l in range(Layer):
        W.append([])
        for k in range(User):
            W[l].append( np.matrix( np.zeros((d,d))+np.zeros((d,d))*1j,dtype=np.complex ) )
    
    V=[]
    for l in range(Layer):
        V.append([])
        for k in range(User):
            V[l].append( np.matrix( np.zeros((Nt,d))+np.zeros((Nt,d))*1j,dtype=np.complex ) )
            
    V1=[]
    for l in range(Layer):
        V1.append([])
        for k in range(User):
            V1[l].append( np.matrix( np.zeros((Nt,d))+np.zeros((Nt,d))*1j,dtype=np.complex ) )
    
    
    
    HH = np.matrix( np.zeros((User*Nr,Nt)) + np.zeros((User*Nr,Nt))*1j,dtype=np.complex )
    VV = np.matrix( np.zeros((Nt,User*d)) + np.zeros((Nt,User*d))*1j,dtype=np.complex )
    for k in range(User):
        HH[k*Nr:(k+1)*Nr,:]=H[k]
    
    VV=HH.H*(HH*HH.H).I
    for k in range(User):
        V[0][k]=VV[:,k*d:(k+1)*d]
        
        
    for k in range(User):
        V1[0][k]=V[0][k]#*np.sqrt(Pt)/np.sqrt(temp_trace)
    
    
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
 
    
    G_UH_X=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
    G_UH_Y=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_UH_Z=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_UH_O=[ [ np.matrix(np.zeros((Nr,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    G_WH_X=[ [ np.matrix(np.zeros((d,d),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
    G_WH_Y=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_WH_Z=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    G_VH_X=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
    G_VH_Y=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_VH_Z=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    G_VH_O=[ [ np.matrix(np.zeros((Nt,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
    
    
    forward()
    back_propagation()
    
# Update the gradients of trainable parameters X Y Z O
    for l in range(1,Layer):       
        for k in range(User):
            G_UH_X_batch[l][k] = G_UH_X_batch[l][k] + G_UH_X[l][k]/batch
#            G_UH_Y_batch[l][k] = G_UH_Y_batch[l][k] + G_UH_Y[l][k]/batch
            G_UH_Z_batch[l][k] = G_UH_Z_batch[l][k] + G_UH_Z[l][k]/batch
            G_UH_O_batch[l][k] = G_UH_O_batch[l][k] + G_UH_O[l][k]/batch
            
            G_WH_X_batch[l][k] = G_WH_X_batch[l][k] + G_WH_X[l][k]/batch
#            G_WH_Y_batch[l][k] = G_WH_Y_batch[l][k] + G_WH_Y[l][k]/batch
            G_WH_Z_batch[l][k] = G_WH_Z_batch[l][k] + G_WH_Z[l][k]/batch
            
            G_VH_X_batch[l][k] = G_VH_X_batch[l][k] + G_VH_X[l][k]/batch
#            G_VH_Y_batch[l][k] = G_VH_Y_batch[l][k] + G_VH_Y[l][k]/batch
            G_VH_Z_batch[l][k] = G_VH_Z_batch[l][k] + G_VH_Z[l][k]/batch
            G_VH_O_batch[l][k] = G_VH_O_batch[l][k] + G_VH_O[l][k]/batch

# %%        
    if ((i+1) % batch==0 ):
################################### Test the performance of current trainable parameters 
        for ii in range(100):
            H_sample=[]
            for k in range(User):
                H_sample.append( np.matrix( np.random.randn(Nr,Nt) + np.random.randn(Nr,Nt)*1j,dtype=np.complex ) )
                
            V_sample=[]
            for l in range(Layer):
                V_sample.append([])
                for k in range(User):
                    V_sample[l].append( np.matrix( np.zeros((Nt,d))+np.zeros((Nt,d))*1j,dtype=np.complex ) )
                
            HH_sample = np.matrix( np.zeros((User*Nr,Nt)) + np.zeros((User*Nr,Nt))*1j,dtype=np.complex )
            VV_sample = np.matrix( np.zeros((Nt,User*d)) + np.zeros((Nt,User*d))*1j,dtype=np.complex )
            for k in range(User):
                HH_sample[k*Nr:(k+1)*Nr,:]=H_sample[k]
            
            VV_sample = HH_sample.H*(HH_sample*HH_sample.H).I
            for k in range(User):
                V_sample[0][k] = VV_sample[:,k*d:(k+1)*d]
                
            obj_value_temp = obj_value_temp + obj_func.objective_func_whole(delta_noise, X_U, X_W, X_V, Y_U, Y_W, Y_V, Z_U, Z_W, Z_V, O_U, O_V, H_sample, Nr, Nt, d, Pt, sigma, User, Layer, V_sample[0])/50   
 
    
        print(abs(obj_value_temp))
    
        obj_value[count] = obj_value_temp
        obj_value_temp = 0
     
#################################### 
# %%   
################### Update the trainable parameters X Y Z O from Layer-2 to 1  
################### Learning rates need to be selected properly
        count = count+1
        step_size_X = 1/count**(0.6)*0.1
        step_size_Y = 1/count**(0.6)*0
        step_size_Z = 1/count**(0.5)*0.1
        step_size_O = 1/count**(0.5)*0.1
        
    
        for l in Layer_inverse1:       
            for k in range(User):
                X_V[l+1][k] = X_V[l+1][k] + step_size_X*G_VH_X_batch[l+1][k]/np.linalg.norm( G_VH_X_batch[l+1][k],ord=2 ) 
#                Y_V[l+1][k] = Y_V[l+1][k] + step_size_Y*G_VH_Y_batch[l+1][k]/np.linalg.norm( G_VH_Y_batch[l+1][k],ord=2 ) 
                Z_V[l+1][k] = Z_V[l+1][k] + step_size_Z*G_VH_Z_batch[l+1][k]/np.linalg.norm( G_VH_Z_batch[l+1][k],ord=2 ) 
                O_V[l+1][k] = O_V[l+1][k] + step_size_O*G_VH_O_batch[l+1][k]/np.linalg.norm( G_VH_O_batch[l+1][k],ord=2 ) 
                
                X_W[l+1][k] = X_W[l+1][k] + step_size_X*G_WH_X_batch[l+1][k]/np.linalg.norm( G_WH_X_batch[l+1][k],ord=2 ) 
#                Y_W[l+1][k] = Y_W[l+1][k] + step_size_Y*G_WH_Y_batch[l+1][k]/np.linalg.norm( G_WH_Y_batch[l+1][k],ord=2 ) 
                Z_W[l+1][k] = Z_W[l+1][k] + step_size_Z*G_WH_Z_batch[l+1][k]/np.linalg.norm( G_WH_Z_batch[l+1][k],ord=2 ) 
                 
                X_U[l+1][k] = X_U[l+1][k] + step_size_X*G_UH_X_batch[l+1][k]/np.linalg.norm( G_UH_X_batch[l+1][k],ord=2 ) 
#                Y_U[l+1][k] = Y_U[l+1][k] + step_size_Y*G_UH_Y_batch[l+1][k]/np.linalg.norm( G_UH_Y_batch[l+1][k],ord=2 ) 
                Z_U[l+1][k] = Z_U[l+1][k] + step_size_Z*G_UH_Z_batch[l+1][k]/np.linalg.norm( G_UH_Z_batch[l+1][k],ord=2 )  
                O_U[l+1][k] = O_U[l+1][k] + step_size_O*G_UH_O_batch[l+1][k]/np.linalg.norm( G_UH_O_batch[l+1][k],ord=2 )  
                
##################  Layer-1
        for k in range(User):
            X_W[Layer-1][k] = X_W[Layer-1][k] + step_size_X*G_WH_X_batch[Layer-1][k]/np.linalg.norm( G_WH_X_batch[Layer-1][k],ord=2 ) 
#            Y_W[Layer-1][k] = Y_W[Layer-1][k] + step_size_Y*G_WH_Y_batch[Layer-1][k]/np.linalg.norm( G_WH_Y_batch[Layer-1][k],ord=2 )  
            Z_W[Layer-1][k] = Z_W[Layer-1][k] + step_size_Z*G_WH_Z_batch[Layer-1][k]/np.linalg.norm( G_WH_Z_batch[Layer-1][k],ord=2 ) 
             
            X_U[Layer-1][k] = X_U[Layer-1][k] + step_size_X*G_UH_X_batch[Layer-1][k]/np.linalg.norm( G_UH_X_batch[Layer-1][k],ord=2 ) 
#            Y_U[Layer-1][k] = Y_U[Layer-1][k] + step_size_Y*G_UH_Y_batch[Layer-1][k]/np.linalg.norm( G_UH_Y_batch[Layer-1][k],ord=2 )  
            Z_U[Layer-1][k] = Z_U[Layer-1][k] + step_size_Z*G_UH_Z_batch[Layer-1][k]/np.linalg.norm( G_UH_Z_batch[Layer-1][k],ord=2 )  
            O_U[Layer-1][k] = O_U[Layer-1][k] + step_size_O*G_UH_O_batch[Layer-1][k]/np.linalg.norm( G_UH_O_batch[Layer-1][k],ord=2 )  
            
            
################## Reset the gradients of trainable parameters X Y Z O to be zero
        G_UH_X_batch=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
        G_UH_Y_batch=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
        G_UH_Z_batch=[ [ np.matrix(np.zeros((Nr,Nr),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
        G_UH_O_batch=[ [ np.matrix(np.zeros((Nr,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
        
        G_WH_X_batch=[ [ np.matrix(np.zeros((d,d),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
        G_WH_Y_batch=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
        G_WH_Z_batch=[ [ np.matrix(np.zeros((d,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
        
        G_VH_X_batch=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex))  for i in range(User) ] for j in range(Layer) ]
        G_VH_Y_batch=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
        G_VH_Z_batch=[ [ np.matrix(np.zeros((Nt,Nt),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
        G_VH_O_batch=[ [ np.matrix(np.zeros((Nt,d),dtype=np.complex)) for i in range(User) ] for j in range(Layer) ]
            
#fr.close()

endtime = datetime.datetime.now()
print (endtime - starttime)

#plt.plot(abs(obj_value))
#obj_value_temp = obj_value[0:100]
#plt.plot(abs(obj_value_temp))

####################### Store the weights 
#fw = open('../Store_weights/weights.pkl','wb')   
#pickle.dump(X_V, fw)   
#pickle.dump(Y_V, fw) 
#pickle.dump(Z_V, fw) 
#pickle.dump(O_V, fw) 
# 
#pickle.dump(X_W, fw)   
#pickle.dump(Y_W, fw) 
#pickle.dump(Z_W, fw) 
#
#pickle.dump(X_U, fw)   
#pickle.dump(Y_U, fw) 
#pickle.dump(Z_U, fw) 
#pickle.dump(O_U, fw)  
#
#fw.close()  

