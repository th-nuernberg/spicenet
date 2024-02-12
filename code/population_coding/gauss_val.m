% compute the Gaussian tunning curve intersection with a given value
function fval = gauss_val(val, pref, sigma, scale)
     fval = scale*exp(-(val - pref)^2/(2*sigma^2));
end