clc
clear all;
Nt = 64;%The number of transmit antenna
Nr = 2;%The number of receive antenna
K = 30;%The number of users
snr_dB = 30; %SNR in dB
Ptot = 10^(0.1*snr_dB);
sigma2=1;

time_wmmse=0;
tol = 1e-6;
totIter = 3000;
csv_H = csvread('E:\DeepUnfolding_WMMSE\Blackbox CNN\DataSet\Test_H_unsup.csv');

csv_H = csv_H(:,Nr*K*Nr*K+1:end);

size_H = size(csv_H);
length_H = size_H(1);
matrix_H = size_H(2)-2*K;

predict = csvread('E:\DeepUnfolding_WMMSE\Blackbox CNN\DataSet\Predict_UW_unsup.csv');
true_label = csvread('E:\DeepUnfolding_WMMSE\Blackbox CNN\DataSet\True_UW_unsup.csv');

sum_cnn = 0;
sum_zf = 0;
sum_wmmse = 0;

time_zf = 0;
sum = 0;
obj = 0;

bb=zeros(length_H,1);

for i = 1:length_H
    real_HH = csv_H(i,1:matrix_H/2);
    imag_HH = csv_H(i,matrix_H/2+1:matrix_H);
    alpha = csv_H(i,matrix_H+1:matrix_H+K);
    index_list = csv_H(i,end-K+1:end);
    real_HH = reshape(real_HH,Nt,Nr*K)';
    imag_HH = reshape(imag_HH,Nt,Nr*K)';
    HH = complex(real_HH,imag_HH);
    
    for k=1:K
        H(:,:,k) = HH(2*k-1:2*k,:);
        V_mrt(:,:,k) = H(:,:,k)';
    end
    t1=clock;
    Vzf = HH'/(HH*HH');    
    V = reshape(Vzf,Nt,Nr,K);
    Vzf = V;
    
    for k = 1:K
        if index_list(k) == 0
            Hk =  H(:,:,k);
            v1 = Vzf(:,1,k);
            v2 = Vzf(:,2,k);
            v1 = v1/norm(v1);
            v2 = v2/norm(v2);
            value_v = [v1,v2];
            value = [v1'*Hk'*Hk*v1,v2'*Hk'*Hk*v2];
            [~,id_max] = max(value);
            Vzf(:,1,k) = value_v(:,id_max);
            Vzf(:,2,k) = 0;
        end
    end
    
    V = Vzf;
    p = 0;
    for k=1:K
        V_k = V(:,:,k);
        p = p + norm(V_k, 'fro')^2;
    end
    for k=1:K
        V(:,:,k) = sqrt(Ptot/p)*V(:,:,k);
    end
    
    obj_zf = compute_obj(H, V, sigma2,Ptot);
    t2=clock;
    t_zf = etime(t2,t1);
    time_zf = time_zf+t_zf;
    V0 = V;
    
    t1=clock;
    [obj_R_WMMSE_index_vec, V_R_WMMSE_index, U_R_WMMSE_index,W_R_WMMSE_index, UU_R_WMMSE_index,WW_R_WMMSE_index] = R_WMMSE_index(H, Ptot, tol, totIter, V0, HH, alpha, index_list);
    obj_R_WMMSE_index = obj_R_WMMSE_index_vec(end);
    obj_R_WMMSE = compute_obj(H, V_R_WMMSE_index, sigma2,Ptot);
    
    t2=clock;
    t = etime(t2,t1);
    time_wmmse = time_wmmse+t;
    true_UW = true_label(i,:);
    [obj_R_WMMSE_index] = true_R_WMMSE(H, Ptot, HH,true_UW,K,alpha);  
    UW = predict(i,:);
    UU = UW(1:8*K);
    real_UU = UU(1:end/2);
    imag_UU = UU(end/2+1:end);
    real_UU = reshape(real_UU,2,2*K)';
    imag_UU = reshape(imag_UU,2,2*K)';
    UU = complex(real_UU,imag_UU);
    WW = UW(8*K+1:end);
    real_WW = WW(1:3*K);
    imag_WW = WW(3*K+1:end);
    for k=1:K
        real_Wk = real_WW(3*k-2:3*k);
        imag_Wk = imag_WW(k);
        W(:,:,k) = complex( [real_Wk(1)  real_Wk(2);real_Wk(2)  real_Wk(3)], [0  imag_Wk;-imag_Wk  0]);
    end
    
    for k=1:K
        U(:,:,k) = UU(2*k-1:2*k,:);
    end
        
    A = zeros(2*K,2*K);
    for j=1:K
        HHU = HH * H(:,:,j)'* U(:,:,j);
        A = A + HHU * alpha(j) * W(:,:,j) * HHU'+ alpha(j) * trace(U(:,:,j)*W(:,:,j)*U(:,:,j)')*(HH * HH');
    end

    for k=1:K
        X(:,:,k) = A\(HH*H(:,:,k)'*U(:,:,k)*alpha(k)* W(:,:,k));
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
    
    obj_predict = compute_obj(H, V, sigma2,Ptot)
    
    [obj_zf obj_R_WMMSE_index obj_predict];
    [obj_predict/obj_R_WMMSE_index];
    %sum = sum + obj_predict;
    %bb(i)=obj_predict;
    
    sum_cnn = sum_cnn + obj_predict;
    sum_zf = sum_zf + obj_zf;
    sum_wmmse = sum_wmmse + obj_R_WMMSE_index;
   
end

sum_cnn/length_H   %sum-rate achieved by black-box based CNN
sum_zf/length_H   %sum-rate achieved by zero-forcing
sum_wmmse/length_H   %sum-rate achieved by Reduced WMMSE

