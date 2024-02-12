%% Poisson Noisy Pattern of activity for Input Neurons
%   This function generates a Poisson Noisy pattern of activity
%   for each neuron in population as Initial tuning curve 
%   Variables and Argument Description:
%   x:      input analog value subjected to be decode into ...
%           ... activity pattern, dimention in (Radians)
%   sig:    Defines the spread in radians (Rad)
%   v:      spontaneous level of activity for each neuron (Hz)
%   N:      number of neurons in the population
%   C:      A factor in sec, shows the time duration in which the ...
%           activity of neurons has been considered
function R = tuning_curve_init_2(x, sig, v, N, C, min_p, max_p, noise)

K = 20; % in Hz
P = linspace(min_p,max_p,N);
R = zeros(1,N);     % Pattern of activity, or output tuning curve

for j = 1:1:N
    dis = min( abs(P(j)-x) , max_p - abs(P(j)-x) - min_p); %wrapped arounf distance
%     dis = abs(x-P(j));
    temp = exp( (-dis^2) / (2*sig^2) );
    % fj is the lamda value for poisson Neuron j
    fj = C * (K*temp + v); 
    if(noise ~= 0)
        R(j) = poissrnd(fj);
    else
        R(j) = fj;
    end
end

    
    
    
    