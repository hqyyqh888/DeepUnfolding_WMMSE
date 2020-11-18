% calculate obj
function obj = compute_obj(H, V, sigma2,Ptot)
K=4;
[Nr, Nt, K] = size(H);
alpha=ones(1,K);
% d = size(V, 2);
d = Nr;
obj = 0;
VV = zeros(Nt, Nt);
if length(size(V))>2
    for k=1:K
        VV = VV+V(:,:,k)*V(:,:,k)';
    end
    for k=1:K
        Hk = H(:,:,k);
        Vk = V(:,:,k);
        Jk = eye(Nr)*Ptot + Hk*(VV-Vk*Vk')*Hk';
        HVk = Hk*Vk;
        obj = obj+alpha(k)*log(det(eye(d)+HVk'*inv(Jk)*HVk));
    end
    obj = real(obj);
else
    for k=1:K
        VV = VV+V(:,k)*V(:,k)';
    end
    for k=1:K
        Hk = H(:,:,k);
        Vk = V(:,k);
        Jk = eye(Nr)*Ptot + Hk*(VV-Vk*Vk')*Hk';
        HVk = Hk*Vk;
        obj = obj+alpha(k)*log(det(eye(d)+HVk * HVk' *inv(Jk)));
    end
    obj = real(obj);
end
end
