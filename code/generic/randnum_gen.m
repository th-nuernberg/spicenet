% function to generate non-uniform distributed values in a given interval
% [vmin, vmax] for a powerlaw function
function y = randnum_gen(rtype, vrange, num_vals, dtype)
% exponent of powerlaw distribution of values
expn = 3;
% create a uniformly distributed vector of values
x = rand(num_vals, 1)*vrange;
% init bounds
vmin = 0.0; vmax = vrange;
% check if uniform / non-uniform distribution
switch rtype
    case 'uniform'
        y  = -vrange + rand(num_vals, 1)*(2*vrange);
    case 'non-uniform'
        switch dtype
            case 'incpowerlaw'
                y  = exp(log(x*(-vmin^(expn+1) + vmax^(expn+1)) + vmin^(expn+1))/(expn+1));
            case 'decpowerlaw'
                y  = -exp(log(x*(-vmin^(expn+1) + vmax^(expn+1)) + vmin^(expn+1))/(expn+1));
            case 'gauss'
                y  = randn(num_vals, 1)*(vmax/4);
            case 'convex'
                y = zeros(num_vals, 1);
                for idx = 1:length(x)
                    if idx<=length(x)/2
                        vmin = 0.00000005;
                        y(idx)  = -exp(log(x(idx)*(-vmin^(expn+1) + vmax^(expn+1)) + vmin^(expn+1))/(expn+1));
                    else
                        vmin = -vmin;
                        y(idx)  = exp(log(x(idx)*(-vmin^(expn+1) + vmax^(expn+1)) + vmin^(expn+1))/(expn+1));
                    end
                end
        end
end