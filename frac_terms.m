close all;
% function gen= get_frac_term_gen(order)
%     gen = @(h) 
% end

function term = frac_term(order,h)
    term = (-1).^h.*gamma(order + 1)./(gamma(h+1) .*gamma(order - h +1));
end

% fsurf(@(x,y) log10(abs(frac_term(x,y))),[1 2 0 20]);
% xlabel("order");
% ylabel("term");
% zlabel("abs value ")
% title("Log graph for term values");

figure();
% fplot(@(h) abs(frac_term(1.16,h)),[0 20]);
plot(abs(frac_term(1.16,0:20)),'*');
title("Terms for a given order");
xlabel("Terms");

figure();
% fplot(@(h) log(abs(frac_term(1.16,h))),[0 20]);
plot(log(abs(frac_term(1.16,0:20))),'*');
title("Log Terms for a given order");
xlabel("Terms");

figure();
fplot(@(order) abs(frac_term(order,0)),[1 2]);
title("Values for term 0");
xlabel("orders");

figure();
fplot(@(order) abs(frac_term(order,1)),[1 2]);
title("Values for term 1");
xlabel("orders");

figure();
fplot(@(order) abs(frac_term(order,2)),[1 2]);
title("Values for term 2");
xlabel("orders");

figure();
fplot(@(order) abs(frac_term(order,3)),[1 2]);
title("Values for term 3");
xlabel("orders");