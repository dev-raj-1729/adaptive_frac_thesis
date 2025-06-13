function out= iso_frac_filter(v,n)
    m = (n-1)/2;
    weights = frac_term(v,0:m);
    weights = [flip(weights),weights(2:end)];
    out = diag(weights);
    out = out + flip(out);
    out(m+1,:) = weights;
    out(:,m+1) = weights;
end