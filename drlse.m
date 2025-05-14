% Clean DRLSE Implementation Example in MATLAB
clear; close all; clc;

%% Load and preprocess image
% I = imread('coins.png');
% size(I)
% I = double(I);
% I = imgaussfilt(I, 1);  % Smooth image to reduce noise
% I = (I - min(I(:))) / (max(I(:)) - min(I(:)));  % Normalize to [0, 1]
I = zeros([700,700]);
I(300:400,300:400) = 1;

%% Parameters
mu = 0.2;         % distance regularization term
lambda = 5;       % weight of the length term (edge attraction)
alpha = -0.5;       % balloon force
epsilon = 1.5;    % parameter for Dirac and Heaviside approximation
timestep = 1;     % time step
iter = 2000;       % number of iterations

%% Create edge indicator function
[Gx, Gy] = gradient(I);
g = 1 ./ (1 + Gx.^2 + Gy.^2);

%% Initialize level set function (LSF) as binary step function
c0 = 2;
phi = c0 * ones(size(I));
phi(30:100, 30:100) = -c0;  % Initial contour inside rectangle

%% Evolution loop
for n = 1:iter
    phi = drlse_evolution(phi, g, lambda, mu, alpha, epsilon, timestep);

    if mod(n, 20) == 0
        imshow(I, []); hold on;
        contour(phi, [0 0], 'r', 'LineWidth', 2);
        title(['Iteration ', num2str(n)]);
        drawnow;
    end
end

%% Display final result
figure;
imshow(I, []); hold on;
contour(phi, [0 0], 'g', 'LineWidth', 2);
title('Final Contour');

function phi = drlse_evolution(phi, g, lambda, mu, alpha, epsilon, timestep)
    % Compute derivatives
    [phix, phiy] = gradient(phi);
    s = sqrt(phix.^2 + phiy.^2 + 1e-10);
    
    % Normalized gradient
    nx = phix ./ s;
    ny = phiy ./ s;
    
    % Curvature term: div(g * n)
    [gnx, ~] = gradient(g .* nx);
    [~, gny] = gradient(g .* ny);
    edge_term = g .* (divergence(nx, ny)) + gnx + gny;
    
    % Dirac delta approximation
    dirac_phi = (epsilon / pi) ./ (epsilon^2 + phi.^2);
    
    % Energy functional gradient descent
    phi = phi + timestep * (mu * dist_reg_p2(phi) + lambda * dirac_phi .* edge_term + alpha * dirac_phi .* g);
end

function f = dist_reg_p2(phi)
    % Double-well potential derivative: (|grad(phi)| - 1)
    [phi_x, phi_y] = gradient(phi);
    s = sqrt(phi_x.^2 + phi_y.^2 + 1e-10);
    a = (s >= 0) & (s <= 1);
    b = (s > 1);
    ps = a .* sin(2 * pi * s) / (2 * pi) + b .* (s - 1);
    dps = ps ./ (s + 1e-10);
    
    div = divergence(dps .* phi_x - phi_x, dps .* phi_y - phi_y);
    f = div;
end

function divF = divergence(Fx, Fy)
    [dFxdx, ~] = gradient(Fx);
    [~, dFydy] = gradient(Fy);
    divF = dFxdx + dFydy;
end
