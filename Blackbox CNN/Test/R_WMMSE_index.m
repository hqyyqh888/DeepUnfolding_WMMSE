%in this code, X denote the digital transmit pecoder, i.e., Vbb
function [obj_vec, V, U, W, UU, WW, UWU, UUWW] = R_WMMSE_index(H, Ptot, tol, totIter, V0, HH,alpha, index_list)
[Nr, Nt, K] = size(H);
d = size(V0, 2);

V = V0;
sigma2=1;
maxIter = totIter;
obj = Inf;
obj_vec = [];
iter = 0;
while(iter<maxIter)
    iter = iter+1;
    obj_old = obj;
    
    VV_T = zeros(Nt, Nt);
    for k=1:K
        V_k = V(:,:,k);
        VV_T = VV_T+V_k*V_k';
    end
   
    gamma = real(trace(VV_T));
    obj = 0;
  
    A = zeros(Nr*K,Nr*K);
    UU = [];
    WW = [];
    
    for k=1:K
        H_k = H(:,:,k);
        if H_k(2,:) == 0 
            H_k = H_k(1,:);
            HkVVHk = H_k*VV_T*H_k';
            V_k = V(:,:,k);
            V_k = V_k(:,1);
            U_k = (gamma*eye(1)+HkVVHk)\(H_k*V_k);
            W_k = 1/real(1-U_k'* H_k * V_k);
            
            HHU = HH * H_k'* U_k;
            A = A + HHU * alpha(k) * W_k * HHU'+ alpha(k) * trace(U_k*W_k*U_k')*(HH * HH');
            
            U_k = [U_k,0;0,0];
            U(:,:,k) = U_k;
            W(:,:,k) = complex( [W_k  0;0  0], [0  0;0  0]);

        else
            HkVVHk = H_k*VV_T*H_k';
            V_k = V(:,:,k);
            % update U
            if index_list(k) == 0
                V_k = V_k(:,1);
                U_k = (gamma*eye(Nr)+HkVVHk)\(H_k*V_k);
                W_k = 1/real(1-U_k'* H_k * V_k);
                
                HHU = HH * H_k'* U_k;
                A = A + HHU * alpha(k) * W_k * HHU'+ alpha(k) * trace(U_k*W_k*U_k')*(HH * HH');
                
                U_k = [U_k,zeros(2,1)];
                U(:,:,k) = U_k;
                W(:,:,k) = complex( [W_k  0;0  0], [0  0;0  0]);
            else
                U_k = (gamma*eye(Nr)+HkVVHk)\(H_k*V_k);
                U(:,:,k) = U_k;
                
                % update W
                W_k = inv(eye(d) - U_k'* H_k * V_k);
                W(:,:,k) = 0.5*(W_k+W_k');
                
                HHU = HH * H_k'* U_k;
                A = A + HHU * alpha(k) * W_k * HHU'+ alpha(k) * trace(U_k*W_k*U_k')*(HH * HH');
            end
        end
        
        
        UU = [UU;U(:,:,k)];
        WW = [WW;W(:,:,k)];
        
        obj = obj + alpha(k)*log(det(W_k));

    end
    
    obj = real(obj);
    obj_vec = [obj_vec obj];
    
    if  abs(obj-obj_old)/abs(obj_old)<=tol
        break;
    end
    
    XX = [];
    for k=1:K
        H_k = H(:,:,k);
        if H_k(2,:) == 0 
            H_k = H_k(1,:);
            U_k = U(:,:,k);
            U_k = U_k(1);
            W_k = W(:,:,k);
            W_k = W_k(1);
            X_k = pinv(A)*(HH*H_k'*U_k*alpha(k)*W_k);
            V_k = HH'* X_k;
            V(:,:,k) = [V_k,zeros(Nt,1)];
        else
            if index_list(k) == 0
                U_k = U(:,:,k);
                U_k = U_k(:,1);
                W_k = W(:,:,k);
                W_k = W_k(1);
                X_k = pinv(A)*(HH*H_k'*U_k*alpha(k)*W_k);
                V_k = HH'* X_k;
                V(:,:,k) = [V_k,zeros(Nt,1)];
            else
                X(:,:,k) = pinv(A)*(HH*H_k'*U(:,:,k)*alpha(k)* W(:,:,k));
                V(:,:,k) = HH'* X(:,:,k);
                XX = [XX X(:,:,k)];
            end
        end
    end

    UUWW = [];
    for k=1:K
    	UWU(:,:,k) = U(:,:,k)*W(:,:,k)*U(:,:,k)';
    	UW(:,:,k) = U(:,:,k)*W(:,:,k);
        UUWW = [UUWW;UW(:,:,k)];
    end
    
end

p = 0;
for k=1:K
    V_k = V(:,:,k);
    p = p + norm(V_k, 'fro')^2;
end
for k=1:K
    V(:,:,k) = sqrt(Ptot/p)*V(:,:,k);
end

compute_obj(H, V, sigma2,Ptot);
end

