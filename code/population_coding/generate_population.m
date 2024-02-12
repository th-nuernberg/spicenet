% function that generates a population of neurons with given properties
% returns a struct array 

% neuron information (neuron index i)
%   i       - index in the population 
%   ri      - activity of the neuron (firing rate)
%   fi      - tuning curve (e.g. Gaussian)
%   vi      - preferred value - even distribution in the range
%   etai    - neuronal noise value - zero mean and tipically correlated

% population information 
%   size            - number of neurons in the population 
%   range           - the range of values of the population representation
%   sigma           - population standard deviation -> coarse (big val) / sharp (small val) receptive field
%   eta             - neuronal noise
%   eta_scale       - noise scale of neuronal noise
%   min_rate        - minimum firing rate of neurons (background rate)
%   max_rate        - maximum firing rate (normalization)

function y = generate_population(size, range, sigma, eta_scale, min_rate, max_rate)
% for representation increment size
size = size + 1;
% zero mean neuronal noise
eta = randn(size, 1)*eta_scale;
% init population
for idx=1:size
    % evenly distributed preferred values in the interval
    vi = - range + (idx-1)*(range/((size-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(vi, ...
                               sigma, ...
                               range, ...
                               max_rate);
    % tuning curves setup
    fi.p = pts;
    fi.v = vals;
    % population wrap up
    y(idx) = struct('i',    idx, ...
                    'vi',   vi,...
                    'fi',   fi, ...
                    'eta', eta(idx),...
                    'ri',   abs(randi([min_rate , max_rate])),...
                    'size', size,...
                    'range', range, ...
                    'sigma', sigma, ...
                    'eta_scale', eta_scale,...
                    'min_rate', min_rate, ...
                    'max_rate', max_rate);
end;
end