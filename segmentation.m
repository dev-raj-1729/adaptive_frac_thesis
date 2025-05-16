clear all;
close all;
%Helper function for next figure count
global figcount;
figcount = 0;
function c = nfc()
    global figcount;
    figcount = figcount + 1;
    c = figcount;
end

% Control parameters
mu = 0.0002; % Weight for reg term
nu = 1; % weight for line term
lambda = 20; % weight for fit term
gam = 0.2; % weight for fractional fit term

delta_t = 1;
sigma = 3;
epsilon = 1.5; % smoothing For dirac delta and heaviside
alpha = 1;
beta = 0.8;
no_iter = 10000;
out_every = 1000; % no iteration after which plot is updated

% function out = DiracDelta(x,epsilon)
%     out = (abs(x) < epsilon).*((1 + cos(pi* x / epsilon))/(2*epsilon));
% end
% figure(nfc());
% fplot(@(x) DiracDelta(x,epsilon));
% title("Dirac Delta")


% Continuous Heaviside Function
% function out = Heaviside(x,epsilon)
%     out = (abs(x) < epsilon) .* (0.5*(1 + x/epsilon + (1/pi)*sin(pi*x/epsilon))) +...
%         (x > epsilon);
% end

%% Heaviside function 
function h = Heaviside(x,e)
   h =  0.5 .*(1 + (2*atan(x/e))/pi);
end

%% Dirac Delta function
function d = DiracDelta(x,e)
    d = e./(pi*(e^2 + x.^2));
end

% I = rgb2gray(imread("drlse_knee.jpg"));
I = imread("inputs/oval_and_rectangle.png");
I = double(I)/255; % Normalize Image

gauss_filter = fspecial('gaussian',6*ceil(sigma)+1,sigma);
% I = conv2(I,gauss_filter,"same");

[grad_I,grad_dir] = imgradient(I);
% Calculate Order Matrix 
P = (255*grad_I + alpha)./ (255*grad_I + beta);

tic
I_frac = FracDerConv(I,P);
toc
writematrix(I_frac,"./matrices/conv_frac.csv");
figure(nfc());
imshow(I_frac);
title("Convolution Fractional Derivative");

%============================================
% Calculate edge stop function 
I_frac_smooth = conv2(I_frac,gauss_filter,"same");
% edgeStop = 1./(1 + I_frac_smooth);
edgeStop = 1./(1 + I_frac_smooth);
figure(nfc());
imshow(edgeStop);
title("Edge Stop Function");

% ========
% Initialize Level Set
phi =  2*ones(size(I));
phi(32:224,32:224) = -2;
figure(nfc());
imshow(I,[]);hold on;
contour(phi, [0 0], 'r', 'LineWidth', 1);
title("Initial Level set");

figure(nfc());

line_terms = zeros([1,no_iter]);
area_terms = zeros([1,no_iter]);
reg_terms = zeros([1,no_iter]);
fit_terms = zeros([1,no_iter]);
frac_fit_terms = zeros([1,no_iter]);

K = gauss_filter;
KI = imfilter(I,K,'replicate');
FI = I + I_frac;
FKI = imfilter(FI,K,"replicate");

tic
for iter = 1:no_iter
    phi = NeumannBoundCond(phi);
    [phi_x,phi_y] = gradient(phi);
    grad_phi = sqrt(phi_x.^2 + phi_y.^2) + 1e-10;
    delta_phi = DiracDelta(phi,epsilon);
    H_phi = Heaviside(phi,epsilon);
    %% Regularization
    dr_phi = dr(grad_phi);
    reg_term = divergence(dr_phi.*phi_x,dr_phi.*phi_y);
    reg_terms(iter) = mu*max(abs(reg_term),[],"all");
    %% Line Term
    div_gphi = divergence(edgeStop.*phi_x ./ grad_phi,...
                        edgeStop.*phi_y./ grad_phi);
    line_term = delta_phi.*div_gphi;
    line_terms(iter) = nu*max(abs(line_term),[],"all");
    %% fit term
    KIH_1 = imfilter(I.*H_phi,K,"replicate");
    KH_1 = imfilter(H_phi,K,"replicate");
    f1 = KIH_1 ./ KH_1;

    KIH_2 = KI - KIH_1;
    KH_2 = 1 - KH_1;
    f2 = KIH_2 ./ KH_2;

    % R1 is zero because lambda_1 = lambda_2
    R2 = I.*imfilter(f1 - f2,K,"replicate");
    R3 = imfilter(f1.^2 - f2.^2,K,"replicate");
    fit_term = -delta_phi.*( R3 - 2*R2);
    fit_terms(iter) = lambda*max(abs(fit_term),[],"all");
    %% Fractional Fit Terms
    FKIH_1 = imfilter(I.*H_phi,K,"replicate");
    FKIH_2 = FKI - FKIH_1;
    b1 = FKIH_1 ./ KH_1;
    b2 = FKIH_2 ./ KH_2;

    R2 = FI.*imfilter(b1 - b2,K,"replicate");
    R3 = imfilter(b1.^2 - b2.^2,K,"replicate");
    frac_fit_term = -delta_phi.*(R3 - 2*R2);
    frac_fit_terms(iter) = gam*max(abs(frac_fit_term),[],"all");
    
    %% Updation
    L = mu*reg_term + nu*line_term + lambda*fit_term ...
        +gam*frac_fit_term;

    phi = phi + delta_t*L;
    if mod(iter, out_every) == 0
        imshow(I, []); hold on;
        contour(phi, [0 0], 'r', 'LineWidth', 1);
        title(['Iteration ', num2str(iter)]);
        drawnow;
    end
end
toc
% Iteration End
% figure(nfc());
% imshow((phi<0).*I);
% surf(phi);
% title("segmentation");
writematrix(phi,"./matrices/phi.csv");

figure(nfc());
% surf(-phi);
imshow(I,[]);hold on;
contour(phi, [0 0], 'r', 'LineWidth', 1);
title("final level set");

% writematrix(line_terms,"./matrices/line_terms.csv");
% writematrix(area_terms,"./matrices/area_terms.csv");

figure(nfc())
plot(line_terms);
title("Line terms");

% figure(nfc());
% plot(area_terms);
% title("area terms");
figure(nfc());
plot(fit_terms);
title("fitting terms");

figure(nfc());
plot(frac_fit_terms);
title("fractional fiting terms");

figure(nfc());
plot(reg_terms);
title("reg terms");

% Calculate Divergence
function d = div(Fx,Fy)
    [Fxx,Fxy] = gradient(Fx);
    [Fyx,Fyy] = gradient(Fy);
    d = Fxx + Fyy;
end

% Iterative Way of calculating fractional Derivative
% Assume it is zero padded
function g = FracDerIterative(I,order_matrix)
    [n,m] = size(I);
    gx = zeros(size(I));
    gy = zeros(size(I));
    for i = 1:n 
        for j = 1:m

            % Calculate Fractional Derivative for a pixel
            order = order_matrix(i,j);
            for h = 0:i-1
                gx(i,j) = gx(i,j) + ((-1)^h) * gamma(order + 1)*I(i-h,j)/(gamma(h+1) * gamma(order - h + 1));
            end

            for k = 0:j-1
                gy(i,j) = gy(i,j) + ((-1)^k) * gamma(order + 1)*I(i,j-k)/(gamma(k+1)* gamma(order - k + 1));
            end
        end
    end
    g = abs(gx) + abs(gy);
end

function g = FracDerConv(I,order_matrix)
    [n,m] = size(I);
    % gx = zeros(I);
    % gy = zeros(I);
    g = zeros(size(I));
    parfor i = 1:n
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

function phi = NeumannBoundCond(phi)
    % Enforce Neumann boundary conditions by copying edge pixels
    [nrow, ncol] = size(phi);

    % Corners
    phi(1,1) = phi(3,3);
    phi(1,ncol) = phi(3,ncol-2);
    phi(nrow,1) = phi(nrow-2,3);
    phi(nrow,ncol) = phi(nrow-2,ncol-2);

    % Top and bottom rows
    phi(1,2:ncol-1) = phi(3,2:ncol-1);
    phi(nrow,2:ncol-1) = phi(nrow-2,2:ncol-1);

    % Left and right columns
    phi(2:nrow-1,1) = phi(2:nrow-1,3);
    phi(2:nrow-1,ncol) = phi(2:nrow-1,ncol-2);
end

function fit = CalculateFitTerm(I,K,KI,H_phi,delta_phi)
    KIH_1 = imfilter(I.*H_phi,K,"replicate");
    KH_1 = imfilter(H_phi,K,"replicate");
    f1 = KIH_1 ./ KH_1;

    KIH_2 = KI - KIH_1;
    KH_2 = 1 - KH_1;
    f2 = KIH_2 ./ KH_2;

    % R1 is zero as lambda_1 = lambda_2 here
    R2 = I.*imfilter(f1-f2,K,"replicate");
    R3 = imfilter(f1.^2 - f2.^2,K,"replicate");
    fit = -delta_phi .*(R3 - 2*R2);
end

function out = dr(x)
    zrs = (x==0);
    out =  zrs + ...
            (x > 0 & x <1).*(sin(2*pi*x)./(2*pi*x + zrs)) + ...
            (x >= 1).*(1 - 1./(x+zrs) );

end