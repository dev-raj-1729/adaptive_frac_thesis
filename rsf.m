close all;
sigma = 3.0; % for gaussian
w = 13; % size of gaussian mask
lambda_1 = 1000.0;
lambda_2 = 1000.0;
delta_t = 0.1;
mu = 0.01;
nu = 5; 
epsilon = 1.5;

c0 = 2; % Initial Contour Value
no_iter = 2000;
out_every = 10000;
small_constant = 1e-10; % for avoiding divsion by zero

% figure();
% fplot(@(x)Heaviside(x,epsilon));
% title("Heaviside");

% figure()
% fplot(@(x)DiracDelta(x,epsilon));
% title("Dirac Delta");

I = imread("inputs/oval_and_rectangle.png");
I = double(I)/255;
phi = -c0*ones(size(I));
phi(32:224,32:224) = c0;
figure();
imshow(I);
title("Initial Level Set");
hold on;
contour(phi,[0 0],'r','LineWidth',1);
drawnow;

K = fspecial('gaussian',w,sigma);
KI = imfilter(I,K,"replicate");
II = I.^2;
R1 = (lambda_1 - lambda_2).*II;

figure();

line_terms = zeros([1 no_iter]);
fit_terms = zeros([1 no_iter]);
reg_terms = zeros([1 no_iter]);

tic
for iter = 1:no_iter
    phi = NeumannBoundCond(phi);
    H_phi = Heaviside(phi,epsilon);
    delta_phi = DiracDelta(phi,epsilon);
    [phi_x,phi_y] = gradient(phi);
    grad_phi = sqrt(phi_x.^2 + phi_y.^2) + small_constant;
    div_phi = divergence(phi_x ./ grad_phi, phi_y./grad_phi);
    lap_phi = 4*del2(phi);

    %% Calculate Fitting Terms
    KIH_1 = imfilter(I.*H_phi,K,"replicate");
    KH_1 = imfilter(H_phi,K,"replicate");
    f1 = KIH_1./KH_1;

    KIH_2 = KI - KIH_1;
    KH_2 = 1 - KH_1;
    f2 = KIH_2 ./KH_2;

    R2 = I.*imfilter(lambda_1*f1 - lambda_2*f2,K,"replicate");
    R3 = imfilter(lambda_1*f1.^2- lambda_2*f2.^2,K,"replicate");
    fitting_term =   -delta_phi.*(R1 - 2*R2 + R3);
    % fit_terms(iter) = max(fitting_term,[],"all");

    %% Line Term 
    line_term = delta_phi.*div_phi;
    % line_terms(iter) = nu*max(line_term,[],"all");
    %% Regularization Term
    reg_term = lap_phi - div_phi;
    % reg_terms(iter) = mu*max(reg_term,[],"all");
    %% Updation
    L = fitting_term  + nu*line_term + mu*reg_term;
    phi = phi + delta_t.*L;

    % if mod(iter,out_every) == 0 
    %     imshow(I); hold on;
    %     contour(phi,[0 0],'r','LineWidth',1);
    %     title(['Iteration ',num2str(iter)]);
    %     drawnow;
    % end
end
toc

figure();
imshow(I); hold on;
contour(phi,[0 0],'r','LineWidth',1);
title("Final Level Set");
hold off;

% figure();
% plot(fit_terms);
% title("Fit terms");

% figure()
% plot(line_terms);
% title("Line Terms");

% figure()
% plot(reg_terms);
% title("Reg Terms");


function d = div(Fx,Fy)
    [Fxx,Fxy] = gradient(Fx);
    [Fyx,Fyy] = gradient(Fy);
    d = Fxx + Fyy;
end

%% Heaviside function 
function h = Heaviside(x,e)
   h =  0.5 .*(1 + (2*atan(x/e))/pi);
end

function d = DiracDelta(x,e)
    d = e./(pi*(e^2 + x.^2));
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