alpha = 1;
beta = 0.8;
gradients = 0:510;

I = rand([720,420]);
test_order = 1.1024;

orders = (gradients + alpha)./(gradients + beta);

max_order = max(orders,[],"all")
min_order = min(orders,[],"all")

order = alpha/beta;

% Convolution kernel 
ksize = max(size(I));
js = 0:ksize-1;
kernel = ((-1).^js) .*(gamma(order + 1)./(gamma(js+1).*gamma(order -js +1)));
gy1 = conv2(I,transpose(kernel),"full");
gy1_size = size(gy1)

% Manual Calculation 
[n,m] = size(I);
gy2 = zeros(size(I));
for i = 1:n 
    for j = 1:m
        order = alpha/beta;
        for h = 0:i-1
            gy2(i,j) = gy2(i,j) + ((-1)^h) * gamma(order + 1)*I(i-h,j)/(gamma(h+1)*gamma(order-h+1));
        end
    end
end 

max_error = max(abs(gy1(1:n,:)-gy2),[],"all")