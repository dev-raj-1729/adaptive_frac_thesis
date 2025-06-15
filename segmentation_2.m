clear all;
close all;

%% Control parameters
mu = 0.0002; % Weight for reg term
nu = 1; % weight for line term
lambda = 20; % weight for fit term
% gam = 0.2; % weight for fractional fit term
gam = 0;

delta_t = 1;
sigma = 3;
epsilon = 1.5; % smoothing For dirac delta and heaviside
alpha = 1;
beta = 0.8;
no_iter = 20000;
out_every = 1000; % no iteration after which plot is updated

%% Heaviside function 
function h = Heaviside(x,e)
   h =  0.5 .*(1 + (2*atan(x/e))/pi);
end

%% Dirac Delta function
function d = DiracDelta(x,e)
    d = e./(pi*(e^2 + x.^2));
end

%% Read Image
% I = rgb2gray(imread("inputs/shaded_oval_and_rectangle.png"));
I = imread("inputs/oval_and_rectangle.png");
I = double(I)/255; % Normalize Image
I = imnoise(I, 'gaussian',0,0.005);

%% Enhancement 

h = iso_frac_filter(0.9,51);
I = rescale(imfilter(I,h,"replicate"));

gauss_filter = fspecial('gaussian',6*ceil(sigma)+1,sigma);
% I = conv2(I,gauss_filter,"same");

[grad_I,grad_dir] = imgradient(I);
% Calculate Order Matrix 
imshow(grad_I);
% P = (255*grad_I + alpha)./ (255*grad_I + beta);
P = calc_order(I,15);

tic
I_frac = rescale(abs(FracDerConv(I,P)));
toc
figure();
imshow(I_frac);

%% Calculate edge stop function 
I_frac_smooth = conv2(I_frac,gauss_filter,"same");
% edgeStop = 1./(1 + I_frac_smooth);
edgeStop = 1./(1 + I_frac_smooth);
figure();
imshow(rescale(edgeStop));
title("Edge Stop Function");

%% Initialize Level Set
phi =  2*ones(size(I));
phi(32:224,32:224) = -2;
figure();
imshow(I,[]);hold on;
contour(phi, [0 0], 'r', 'LineWidth', 1);
title("Initial Level set");

figure();

line_terms = zeros([1,no_iter]);
area_terms = zeros([1,no_iter]);
reg_terms = zeros([1,no_iter]);
fit_terms = zeros([1,no_iter]);
frac_fit_terms = zeros([1,no_iter]);

K = gauss_filter;
KI = imfilter(I,K,'replicate');
FI = I + I_frac;
FKI = imfilter(FI,K,"replicate");

%% Iteration
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


figure();
imshow(I,[]);hold on;
contour(phi, [0 0], 'r', 'LineWidth', 1);
title("final level set");


%% Statistics 
figure()
plot(line_terms);
title("Line terms");

figure();
plot(fit_terms);
title("fitting terms");

figure();
plot(frac_fit_terms);
title("fractional fiting terms");

figure();
plot(reg_terms);
title("reg terms");



%% Helper functions
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
function out = dr(x)
    zrs = (x==0);
    out =  zrs + ...
            (x > 0 & x <1).*(sin(2*pi*x)./(2*pi*x + zrs)) + ...
            (x >= 1).*(1 - 1./(x+zrs) );

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