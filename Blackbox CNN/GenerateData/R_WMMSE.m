%in this code, X denote the digital transmit pecoder, i.e., Vbb
function [obj_vec, V, U, W, UU, WW, UWU, UUWW] = R_WMMSE(H, Ptot, sigma2, tol, totIter, V0, HH, alpha)
[Nr, Nt, K] = size(H);
d = size(V0, 2);
V = V0;
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
  
    for k=1:K
        % update U
        H_k = H(:,:,k);
        HkVVHk = H_k*VV_T*H_k';
        V_K = V(:,:,k);
        U_k = (gamma*eye(Nr)+HkVVHk)\(H_k*V_K);%update Ubbk
        U(:,:,k) = U_k;

        % update W
        W_k = pinv(eye(d) - U_k'* H_k * V_K);
        W(:,:,k) = 0.5*(W_k+W_k');  
        
        obj = obj + alpha(k)*log(det(W_k));
    end
    
    obj = real(obj);
    obj_vec = [obj_vec obj];
    
    if  abs(obj-obj_old)/abs(obj_old)<=tol
        break;
    end

    A = zeros(2*K,2*K);
    for j=1:K
        HHU = HH * H(:,:,j)'* U(:,:,j);
        A = A + HHU * alpha(j) * W(:,:,j) * HHU'+ alpha(j) * trace(U(:,:,j)*W(:,:,j)*U(:,:,j)')*(HH * HH');
    end
    
    XX = [];
    UU = [];
    WW = [];
    for k=1:K
        X(:,:,k) = pinv(A) * (HH*H(:,:,k)'*U(:,:,k)*alpha(k)* W(:,:,k));
        V(:,:,k) = HH'* X(:,:,k);
        XX = [XX X(:,:,k)];
        UU = [UU;U(:,:,k)];
        WW = [WW;W(:,:,k)];
    end
    
    UUWW = [];
    for k=1:K
    	UWU(:,:,k) = U(:,:,k)*W(:,:,k)*U(:,:,k)';
    	UW(:,:,k) = U(:,:,k)*W(:,:,k);
        UUWW = [UUWW;UW(:,:,k)];
    end
end

UU = [];
WW = [];
for k=1:K
    if trace(trace(V(:,:,k)*V(:,:,k)'))<10^-5
        U(:,:,k) = 0;
        W(:,:,k) = eye(2);
        X(:,:,k) = pinv(A)*(HH*H(:,:,k)'*U(:,:,k)*alpha(k)* W(:,:,k));
        V(:,:,k) = HH'* X(:,:,k);
    end
    UU = [UU;U(:,:,k)];
    WW = [WW;W(:,:,k)];
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

