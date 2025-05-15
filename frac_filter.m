function g = FracDerConv(I,order_matrix)
    [n,m] = size(I);
    % gx = zeros(I);
    % gy = zeros(I);
    g = zeros(size(I));
    for i = 1:n
        for j = 1:m
            order = order_matrix(i,j);
            ksize = max([i,j]);
            h = ksize-1: -1 :0;
            kernel = ((-1).^h) .* gamma(order+1)./(gamma(h+1) .* gamma(order-h+1));
            gy = sum(I(1:i,j).*transpose(kernel(ksize-i+1:ksize)),"all");
            gx = sum(I(i,1:j).*kernel(ksize-j+1:ksize),"all");
            g(i,j) = abs(gy) + abs(gx);
        end
    end
end

% Approximate the adaptive fractional derivative by
% grouping them into buckets and choosing a representative order
function g = FracDerApprox(I,order_matrix,no_buckets,a,b)
    buckets = linspace(1,a/b,no_buckets+1);
    ksize = max(size(I));
    [n,m] = size(I);
    % gx = zeros(size(I));
    % gy = zeros(size(I));
    g = zeros(size(I));
    bucket_map = floor(no_buckets * (order_matrix - 1)./(a/b - 1));
    js = 0:ksize-1;
    for bucket = 1:no_buckets 
        start_val = buckets(bucket);
        end_val = buckets(bucket+1);
        rep_order = (start_val + end_val)/2;
        kernel = ((-1).^js) .* (gamma(rep_order)./(gamma(js + 1).* gamma(rep_order - js +1)));
        full_convx = conv2(I,kernel,"full");
        full_convy = conv2(I,transpose(kernel),"full");

        g = g + (bucket_map == bucket) .* (full_convx(1:n,1:m) + full_convy(1:n,1:m));
    end
end