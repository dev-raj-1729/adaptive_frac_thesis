clear figcount;
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
mu = 0.2; % Weight for penalty term
nu = 1.5;
lambda = 1;
gam = 1;

delta_t = 5;
sigma = 3;
epsilon = 1.5; % smoothing For dirac delta and heaviside
alpha = 1;
beta = 0.8;
no_iter = 100;
out_every = 1; % no iteration after which plot is updated

% Continuous Dirac Delta
% syms DiracDeltasym(x)
% DiracDeltasym(x) = piecewise(abs(x) < epsilon,1/(2*epsilon)*(1 + cos(pi*x/epsilon)),0 );
% figure(nfc());
% fplot(DiracDeltasym);
% title("Dirac Delta using symbolic");

function out = DiracDelta(x,epsilon)
    out = (abs(x) < epsilon).*((1 + cos(pi* x / epsilon))/(2*epsilon));
end
% figure(nfc());
% fplot(@(x) DiracDelta(x,epsilon));
% title("Dirac Delta")


% Continuous Heaviside Function
function out = Heaviside(x,epsilon)
    out = (abs(x) < epsilon) .* (0.5*(1 + x/epsilon + (1/pi)*sin(pi*x/epsilon))) +...
        (x > epsilon);
end

% figure(nfc());
% fplot(@(x) Heaviside(x,epsilon));
% title("Heaviside");

% I = rgb2gray(imread("drlse_knee.jpg"));
I = imread("coins.png");
I = double(I)/255; % Normalize Image
% I2 = imread("coins.png");
% writematrix(I2,"coins.csv");


% coins_size = size(I2)
% I = zeros([420,420]);
% I(200:300,200:300) = 1;
gauss_filter = fspecial('gaussian',6*ceil(sigma)+1,sigma);
I = conv2(I,gauss_filter,"same");
% writematrix(gauss_filter,"gaussian.csv");

% figure(nfc());
% imshow(I);
% title("Image");

% Calculate 
% [Ix,Iy] = imgradientxy(I);
% grad_I = abs(Ix) + abs(Iy);
[grad_I,grad_dir] = imgradient(I);
figure(nfc());
imshow(grad_I);
title("Absolute Gradient");
writematrix(grad_I,"./matrices/gradient.csv");

% Calculate Order Matrix 
P = (255*grad_I + alpha)./ (255*grad_I + beta);
writematrix(P,"./matrices/order_matrix.csv");

% Calculate Fractional Derivative
% tic
% I_frac = FracDerIterative(I,P);
% toc
% writematrix(I_frac,"iterative_frac.csv");
% figure(nfc());
% imshow(I_frac);
% title("Iterative Fractional Derivative");

% Calculative Fractional Derivative using convolution
tic
I_frac = FracDerConv(I,P);
toc
writematrix(I_frac,"./matrices/conv_frac.csv");
figure(nfc());
imshow(I_frac);
title("Convolution Fractional Derivative");


% tic 
% I_frac3 = FracDerApprox(I,P,100,alpha,beta);
% toc
% writematrix(I_frac3,"frac_approx.csv");
% figure(nfc());
% imshow(I_frac3./max(I_frac3,[],"all"));
% title("Approximate Fractional Derivative");

% max_error = max(abs(I_frac3-I_frac2),[],"all")
% max_order = max(P,[],"all")
% min_order = min(P,[],"all")

%============================================
% Calculate edge stop function 
I_frac_smooth = conv2(I_frac,gauss_filter,"same");
% edgeStop = 1./(1 + I_frac_smooth);
edgeStop = 1./(1 + 127*I_frac_smooth);
figure(nfc());
imshow(edgeStop);
title("Edge Stop Function");

% =========================
% Define terms for regularization penalty
% syms dr(x)
% dr(x) = piecewise(x==0,1,x<1,sin(2*pi*x)/(2*pi*x),x>=1,1-1/x);
function out = dr(x)
    zrs = (x==0);
    out =  zrs + ...
            (x > 0 & x <1).*(sin(2*pi*x)./(2*pi*x + zrs)) + ...
            (x >= 1).*(1 - 1./(x+zrs) );

end

% figure(nfc());
% fplot(@(x) dr(x),[0,2]);
% title("Double well potential d_R");


% ========
% Initialize Level Set
phi =  2*ones(size(I));
phi(140:210,200:270) = -2;
figure(nfc());
imshow(I,[]);hold on;
contour(phi, [0 0], 'r', 'LineWidth', 1);
title("Initial Level set");

figure(nfc());

line_terms = zeros([1,no_iter]);
area_terms = zeros([1,no_iter]);
pen_terms = zeros([1,no_iter]);
fit_terms = zeros([1,no_iter]);
frac_fit_terms = zeros([1,no_iter]);
tic
for iter = 1:no_iter
    [phi_x,phi_y] = gradient(phi);
    grad_phi = sqrt(phi_x.^2 + phi_y.^2 + 1e-10);

    L = zeros(size(I)); % L(phi_{i,j}^n)
    Delta_phi = DiracDelta(phi,epsilon);
    H_phi = Heaviside(phi,epsilon);

    %% Penalty term calculation i.e mu*div(d_R|grad phi| . grad phi)
    dr_coeff = dr(grad_phi);
    pen_x = dr_coeff.*phi_x;
    pen_y = dr_coeff.*phi_y;
    pen_term = div(pen_x,pen_y);
    pen_terms(iter) = mean(pen_term,"all");
    L = L + mu .*pen_term;

    %% Line term
    coeff = edgeStop ./(grad_phi); 
    line_x = coeff .* phi_x;
    line_y = coeff .* phi_y;
    line_term =  Delta_phi.*div(line_x, line_y ); 
    line_terms(iter) = mean(line_term,"all");
    L = L + nu* line_term ;

    %% Area term 
    % area_term = edgeStop.*DiracDelta(phi,epsilon);
    % area_terms(iter) = mean(area_term,"all");
    % L = L + gam*edgeStop.*DiracDelta(phi,epsilon);

    %% fitting term 
    KIH_1 = imfilter(I.* H_phi,gauss_filter,'replicate');
    KH_1 = imfilter(H_phi,gauss_filter,'replicate') + 1e-10;
    f1 = KIH_1 ./ KH_1;

    KIH_2 = imfilter(I.*(1 - H_phi),gauss_filter,'replicate');
    KH_2 = imfilter(1 - H_phi,gauss_filter,'replicate') + 1e-10;
    f2 = KIH_2 ./ KH_2;

    fitting_term = Delta_phi.*((I - f2).^2 - (I - f1).^2);
    fit_terms(iter) = mean(fitting_term,"all");
    L = L + lambda*fitting_term;

    %% fractional fitting term 
    FI = I + I_frac; 
    
    KFIH_1 = imfilter(FI.*H_phi,gauss_filter,'replicate');
    b1 = KFIH_1 ./ KH_1;

    KFIH_2 = imfilter(FI.*H_phi,gauss_filter,'replicate');
    b2 = KFIH_2 ./ KH_2;

    frac_fit_term = Delta_phi.*((I - b2).^2 - (I - b1).^2);
    frac_fit_term(iter) = mean(frac_fit_term,"all");
    L = L + gam*frac_fit_term;
    %% Updation

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
figure(nfc());
imshow((phi<0).*I);
% surf(phi);
title("segmentation");
writematrix(phi,"./matrices/phi.csv");

figure(nfc());
% surf(-phi);
imshow(I,[]);hold on;
contour(phi, [0 0], 'r', 'LineWidth', 1);
title("final level set");

writematrix(line_terms,"./matrices/line_terms.csv");
writematrix(area_terms,"./matrices/area_terms.csv");

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
plot(pen_terms);
title("pen terms");

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