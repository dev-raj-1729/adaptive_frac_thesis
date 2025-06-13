function orders = calc_order(I,nhood)
    sigma = stdfilt(I,true(nhood));
    max_sigma = max(sigma,[],"all");
    min_sigma = min(sigma,[],"all");

    orders = 1 + sqrt((sigma - min_sigma)./(max_sigma - min_sigma + 1e-10));
end