% topp.m
% 
% This function computes the top-one probabilities of the elements of
% a given vector v

function t = topp(v)
    t = exp(v)/sum(exp(v));
end


function t_prot = topp_prot(u, v)
  t_prot = exp(u)/sum(exp(v));
end