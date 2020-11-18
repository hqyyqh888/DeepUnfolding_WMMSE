% calculate obj
function obj = compute_obj(H, V, sigma2,Ptot)
[Nr, Nt, K] = size(H);
d = size(V, 2);
obj = 0;
VV = zeros(Nt, Nt);
if length(size(V))>2
    for k=1:K
        VV = VV+V(:,:,k)*V(:,:,k)';
    end
    for k=1:K
        Hk = H(:,:,k);
        Vk = V(:,:,k);
        Jk = eye(Nr)*Ptot+ Hk*(VV-Vk*Vk')*Hk';
        HVk = Hk*Vk;
        obj = obj+log(det(eye(d)+HVk'*inv(Jk)*HVk));
    end
    obj = real(obj);
else
    for k=1:K
        VV = VV+V(:,k)*V(:,k)';
    end
    for k=1:K
        Hk = H(:,:,k);
        Vk = V(:,k);
        Jk = eye(Nr)*Ptot+ Hk*(VV-Vk*Vk')*Hk';
        HVk = Hk*Vk;
        obj = obj+log(det(eye(d)+HVk'*inv(Jk)*HVk));
    end
    obj = real(obj);
end
end
