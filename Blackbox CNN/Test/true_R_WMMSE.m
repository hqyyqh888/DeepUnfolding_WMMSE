function [obj_true,U,W] = true_R_WMMSE(H, Ptot, HH,true_UW,K, alpha)
UW = true_UW;
sigma2=1;
%     UU = UW(1:end/2);
UU = UW(1:8*K);
real_UU = UU(1:end/2);
imag_UU = UU(end/2+1:end);
real_UU = reshape(real_UU,2,2*K)';
imag_UU = reshape(imag_UU,2,2*K)';
UU = complex(real_UU,imag_UU);

%     WW = UW(end/2+1:end);
%     real_WW = WW(1:end/2);
%     imag_WW = WW(end/2+1:end);
%     real_WW = reshape(real_WW,2,2*K)';
%     imag_WW = reshape(imag_WW,2,2*K)';
%     WW = complex(real_WW,imag_WW);

WW = UW(8*K+1:end);
real_WW = WW(1:3*K);
imag_WW = WW(3*K+1:end);
for k=1:K
    real_Wk = real_WW(3*k-2:3*k);
    imag_Wk = imag_WW(k);
    %         f2 = real_Wk(3)^2;
    %         a =  real_Wk(1);
    %         if a ~= 0
    %             d = (f2+real_Wk(2)^2+imag_Wk^2)/a;
    %         else
    %             imag_Wk = 0;
    %             real_Wk(2) = 0;
    %             d = 0;
    %         end
    %         W(:,:,k) = complex( [a real_Wk(2);real_Wk(2) d], [0  imag_Wk;-imag_Wk  0]);
    W(:,:,k) = complex( [real_Wk(1)  real_Wk(2);real_Wk(2)  real_Wk(3)], [0  imag_Wk;-imag_Wk  0]);
    %         eig_value = min(eig(W(:,:,k)));
    %         if eig_value<0
    %             num = num+1;
    %         end
end

for k=1:K
    U(:,:,k) = UU(2*k-1:2*k,:);
    %         W(:,:,k) = WW(2*k-1:2*k,:);
end

A = zeros(2*K,2*K);
for j=1:K
    HHU = HH * H(:,:,j)'* U(:,:,j);
    %         A = A + HHU * W(:,:,j) * HHU'+ trace(U(:,:,j)*W(:,:,j)*U(:,:,j)')*(HH * HH');
    A = A + HHU * alpha(j) * W(:,:,j) * HHU'+ alpha(j) * trace(U(:,:,j)*W(:,:,j)*U(:,:,j)')*(HH * HH');
    
end

for k=1:K
    X(:,:,k) = A\(HH*H(:,:,k)'*U(:,:,k)*alpha(k)* W(:,:,k));
    %         X(:,:,k) = A\(HH*H(:,:,k)'*U(:,:,k)*W(:,:,k));
    V(:,:,k) = HH'* X(:,:,k);
end

p = 0;
for k=1:K
    V_k = V(:,:,k);
    p = p + norm(V_k, 'fro')^2;
end
for k=1:K
    V(:,:,k) = sqrt(Ptot/p)*V(:,:,k);
end

obj_true = compute_obj(H, V,sigma2,Ptot);


end
