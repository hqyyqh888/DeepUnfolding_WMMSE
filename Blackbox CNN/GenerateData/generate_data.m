clc
clear all;
Nt = 64;%The number of transmit antennas
Nr = 2;%The number of receive antennas
K = 30;%The number of users
snr_dB = 30; %SNR in dB
Ptot = 10^(0.1*snr_dB);
sigma2=1;

tol = 1e-6;
totIter = 3000;
file_H = [];
file_X = [];
file_V = [];
file_UW = [];
file_UWk = [];
file_UWU = [];
file_pos = [];
iteration = 10000; %Number of samples
obj_ave=0;
t1=clock;

for iter = 1:iteration
    
    if mod(iter,100)==0
        [iter]
    end

    [H, sigma2] = generate_MU_MIMO_channel(K, Nr, Nt, snr_dB, Ptot);
    
    HH = [];
    for k=1:K
        HH = [HH; H(:,:,k)];
        V_mrt(:,:,k) = H(:,:,k)';
    end
  
    index_list = ones(1,K);
    alpha = ones(1,K);
    Vzf = HH'/(HH*HH');    

    V = reshape(Vzf,Nt,Nr,K);
    Vzf = V;
    
    p = 0;
    for k=1:K
        V_k = V(:,:,k);
        p = p + norm(V_k, 'fro')^2;
    end
    for k=1:K
        V(:,:,k) = sqrt(Ptot/p)*V(:,:,k);
    end
    
    obj_zf = compute_obj(H, V, sigma2,Ptot);
    V0 = Vzf;
    
    [obj_R_WMMSE_vec, V_R_WMMSE, U_R_WMMSE, W_R_WMMSE, UU_R_WMMSE, WW_R_WMMSE, ...,
        UWU_R_WMMSE, UUWW_R_WMMSE] = R_WMMSE(H, Ptot, sigma2, tol, totIter, V0, HH, alpha);
    
    obj_R_WMMSE = obj_R_WMMSE_vec(end);
    
    p = 0;
    for k=1:K
        V_k = V_R_WMMSE(:,:,k);
        p = p + norm(V_k, 'fro')^2;
    end
    for k=1:K
        V_R_WMMSE(:,:,k) = sqrt(Ptot/p)*V_R_WMMSE(:,:,k);
    end
    
    obj_single=compute_obj(H, V_R_WMMSE, sigma2,Ptot) ;
    obj_ave=obj_ave+obj_single/iteration;
       
    real_HH = reshape(real(HH)',1,[]);
    imag_HH = reshape(imag(HH)',1,[]);
    
    HH = [];
    for k=1:K
        HH = [HH; H(:,:,k)*sqrt(alpha(k))];
    end
    
    HHT = HH * HH';
    valid_H = [];
    for k=1:2*K
        for j=1:2*K
            if k > j
                valid_H = [valid_H imag(HHT(k,j))];
            else
                valid_H = [valid_H real(HHT(k,j))];
            end         
        end
    end

    file_HH = [valid_H real_HH imag_HH alpha index_list];
    file_H = [file_H;file_HH];
    
    real_UU = reshape(real(UU_R_WMMSE)',1,[]);
    imag_UU = reshape(imag(UU_R_WMMSE)',1,[]);
    file_UU = [real_UU imag_UU];
    
    real_WW = [];
    imag_WW = [];
    for k=1:K
        real_Wk = reshape(real(W_R_WMMSE(:,:,k))',1,[]);
        real_Wk = real_Wk([1,2,4]);
        
        imag_Wk = reshape(imag(W_R_WMMSE(:,:,k))',1,[]);
        imag_Wk = imag_Wk(2);
        
        real_WW = [real_WW real_Wk];
        imag_WW = [imag_WW imag_Wk];
    end

    file_WW = [real_WW imag_WW];
    
    UW = [file_UU file_WW];
    file_UW = [file_UW; UW];
 
end
obj_ave;
filename = 'mat\Input_H.csv';
csvwrite(filename,file_H);

filename = 'mat\Output_UW.csv';
csvwrite(filename,file_UW);

t2=clock;
etime(t2,t1);