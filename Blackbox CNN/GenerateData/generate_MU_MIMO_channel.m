function [H, sigma2] = generate_MU_MIMO_channel(K, Nr, Nt, snr_dB, Ptot)
%d = lambda/2;
HH = [];
for k=1:K
%     d = 0.1+ 0.2*rand;%randomly gerneate user within distance in between [100 300] meter
%     PL = 128.1+37.6*log10(d);%dB
%     PL = 10^(-PL/10);
    H1 = complex(randn(Nr, Nt), randn(Nr, Nt));%*sqrt(PL/2);
%    H1 = H1/norm(H1,'fro');%*sqrt(PL/2);
%    H1 = randn(Nr, Nt)*sqrt(PL/2);
    H(:,:,k) = H1;
    HH = [HH; H1];
end
% snr = 10^(snr_dB/10);
% %compute noise variance
% sigma2 = norm(HH, 'fro')^2 / (K*Nr*snr);
%AvgPower  = 10^(mean(10 * log10(mean(sum(abs(H).^2, 2), 1)), 3)/10);
%sigma2 = AvgPower * 10^(-snr_dB/10);
sigma2 = 1;

H = H/sqrt(sigma2/Ptot);

end