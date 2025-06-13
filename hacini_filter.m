function out = hacini_filter(k,v)

K_D = 10^(-v);
N = floor((40*v + 0.5)/(1-v));
i = 0:N;
temp = 10*v*(1-v);
alpha_i = 1 + 2.10.^((v + i)./temp)
beta_i = alpha_i - 2;
[i_g, j_g] = meshgrid(i,i);
num = prod(1 - 10.^((i_g - j_g + v)./temp))
den = i .* prod(1 - 10.^((i_g - j_g)./temp))
g_i = -K_D .* num ./ den;
k_t = transpose(k);
s_1 = 2*transpose(sum((g_i ./ alpha_i ).*((beta_i ./ alpha_i) .^k_t),2));
s_2 = 2*transpose(sum((g_i ./ alpha_i ).*((beta_i ./ alpha_i) .^(k_t - 1)),2));
out = ( k == 0) .*(K_D) ...
        + ( k > 0) .* (s_1) ... 
        - (k > 1) .*(s_2);
end