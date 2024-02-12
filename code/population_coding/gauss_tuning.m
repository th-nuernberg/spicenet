% Gaussian tuning curve parametrizable for the neurons in the population
function [x, fx]= gauss_tuning(pref, sigma, limit, scale)
    % limit is the range (+/-)
    x = -limit:limit;
    % compute the gaussian
    fx = scale*exp(-(x-pref).^2/(2*sigma*sigma));
end