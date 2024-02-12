%% Demo software usign population code for estimating arbitrary functions
% 
% the setup contains 2 input populations each coding some scalar
% (unimodal) variable which are projected onto a 2D network of units with
% neurons exhibiting short range excitation and long range inhibition
% dynamics
% the ouput from the intermediate layer is projected to an output
% population 
% the network has no explicit input and output as each of the populations
% may be considered inputs / outputs and the processing happens in the
% intermediate layer

%% INITIALIZATION
clear all;
clc; close all;

% define the 1D populations (symmetric (size wise) populations)
POPULATION_SIZE         = 50;
POPULATION_RANGE        = 100;
POPULATION_SIGMA        = 10;
POPULATION_MIN_RATE     = 0;
POPULATION_MAX_RATE     = 100;
POPULATION_NOISE_RATE   = 10;

X_POPULATION_VAL        = 30;
Y_POPULATION_VAL        = 10;
Z_POPULATION_VAL        = 0;

CONVERGENCE_STEPS       = 100;

%% Generate first population and initialize
x_population = generate_population(POPULATION_SIZE, ...
                                   POPULATION_RANGE, ...
                                   POPULATION_SIGMA, ...
                                   POPULATION_NOISE_RATE, ...
                                   POPULATION_MIN_RATE, ...
                                   POPULATION_MAX_RATE);

%% Generate second population and initialize
y_population = generate_population(POPULATION_SIZE, ...
                                   POPULATION_RANGE, ...
                                   POPULATION_SIGMA, ...
                                   POPULATION_NOISE_RATE, ...
                                   POPULATION_MIN_RATE, ...
                                   POPULATION_MAX_RATE);

%% Generate third population and initialize
z_population = generate_population(POPULATION_SIZE, ...
                                   2*POPULATION_RANGE, ...
                                   POPULATION_SIGMA, ...
                                   POPULATION_NOISE_RATE, ...
                                   POPULATION_MIN_RATE, ...
                                   POPULATION_MAX_RATE);
                               
%% VISUALIZATION OF TUNING CURVES AND POPULATION INITIALIZATION
figure;
set(gcf,'color','w');
%% First population visualization
% plot the tunning curves of all neurons for the first population
subplot(6, 4, [1 2]);
x_population = init_population(x_population, X_POPULATION_VAL, 'tuning_curves');
% plot the encoded value in the population
subplot(6, 4, [5 6]);
x_population = init_population(x_population, X_POPULATION_VAL, 'encoded_value');
%% Second population visualization
subplot(6, 4, [3 4]);
y_population = init_population(y_population, Y_POPULATION_VAL, 'tuning_curves');
% plot the encoded value in the population
subplot(6, 4, [7 8]);
y_population = init_population(y_population, Y_POPULATION_VAL, 'encoded_value');
%% Third population visualizaiton
subplot(6, 4, [18 19]);
z_population = init_population(z_population, Z_POPULATION_VAL, 'tuning_curves');
% plot the encoded value in the population
subplot(6, 4, [21 22]);
z_population = init_population(z_population, Z_POPULATION_VAL, 'encoded_value');

%% NETWORK DYNAMICS 
% define a 2D intermediate network layer on which the populations project
% assuming we are projecting the populations x and y and population z will
% encode an arbitrary function phi: z = phi(x, y)

%% FEEDFORWARD NETWORK CONNECTIVITY FROM INPUTS TO INTERMEDIATE LAYER
% connectivity matrix initialization, weights interval
J_MAX = 1;
J_MIN = 0;
% connectivity matrix type {smooth, sharp}
J_MAT_TYPE = 'smooth';

% connectivity matrix random initialization
J =  J_MIN + ((J_MAX - J_MIN)/100).*rand(POPULATION_SIZE+1, POPULATION_SIZE+1);

% connectivity matrix, J, is peaked at i=j
for idx=1:POPULATION_SIZE+1
    for jdx=1:POPULATION_SIZE+1
        % switch profile of weight matrix such that a more smoother
        switch(J_MAT_TYPE)
            case 'smooth'
                % projection in the intermediate layer is obtained - Gauss
                J(idx,jdx) = exp(-((idx-jdx))^2/(2*((POPULATION_SIZE+1)/10)*((POPULATION_SIZE+1)/10)));
            case 'sharp'
                % for linear (sharp) profile of the weight matrix the
                % projection in the intermediate layer is noisier
                if(idx==jdx)
                    J(idx,jdx) = 1;
                end
        end
    end
end

% display the feedforward projection weight matrix
% close all;
% surf(J(1:POPULATION_SIZE+1, 1:POPULATION_SIZE+1));
% return;

% stores the summed input activity for each neuron before intermediate layer
sum_rx = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);
sum_ry = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);

% stores the activity of each neuron in the intermediate layer as a
% superposition of the activities in the input layers
rxy = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);

% compute the total input for each neuron in the intermediate layers
for idx=1:POPULATION_SIZE+1
    for jdx=1:POPULATION_SIZE+1
        % each input population contribution 
        for k = 1:POPULATION_SIZE+1
            sum_rx(idx, jdx) = sum_rx(idx, jdx) + J(idx,k)*x_population(k).ri;
        end
        for l = 1:POPULATION_SIZE+1
            sum_ry(idx, jdx) = sum_ry(idx, jdx) + J(jdx,l)*y_population(l).ri;
        end
        % superimpose contributions from both populations 
        rxy(idx,jdx) = sum_rx(idx, jdx) + sum_ry(idx, jdx);
    end
end

% final activity of a neuron in the intermediate layer 
rxy_normalized = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);

% assemble the intermediate layer and fill in with activity values
for idx  = 1:POPULATION_SIZE+1
    for jdx = 1:POPULATION_SIZE+1
        % compute the activation for each neuron - linear activation
        rxy_normalized = normalize_activity(rxy, ...
                                            POPULATION_MIN_RATE,...
                                            POPULATION_MAX_RATE);
    end
end
%% VISUALIZATION OF INTERMEDIATE LAYER ACTIVITY (ONLY FEEDFORWARD PROP. NORMALIZED)
% intermediate layer activity
h1 = subplot(6, 4, [10 14]);
surf(rxy_normalized(1:POPULATION_SIZE, 1:POPULATION_SIZE));
grid off;
set(gca, 'Box', 'off');  

%% RECURRENT CONNECTIVITY IN THE INTERMEDIATE LAYER AND DYNAMICS
% get rid of the ridges in the activity profile of the intermediate layer
% keep only the bump of activity ("Mexican hat connectivity")

% parameters that control the shape of the W connectivity matrix in the
% intermediate layer 
We      = 30;          % short range excitation strength We > Wi
Wi      = 7;          % long range inhibition strength 
sigma_e = 2;           % excitation Gaussian profile sigma_e < sigma_i
sigma_i = 3;           % inhibiton Gaussian profile
W = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1, POPULATION_SIZE+1, POPULATION_SIZE+1);

% build the recurrent connectivity matrix
for idx=1:POPULATION_SIZE+1
    for jdx = 1:POPULATION_SIZE+1
        for k = 1:POPULATION_SIZE+1
            for l = 1:POPULATION_SIZE+1
                W(idx,jdx,k,l) = We*(exp(-((idx-k)^2+(jdx-l)^2)/(2*sigma_e^2))) - ...
                                 Wi*(exp(-((idx-k)^2+(jdx-l)^2)/(2*sigma_i^2)));
            end
        end
    end
end

%show animated movement of the mexican hat
% figure;
% for i = 1:POPULATION_SIZE+1
%     for j = 1:POPULATION_SIZE+1
%         surf(W(1:POPULATION_SIZE+1, 1:POPULATION_SIZE+1, i, j));
%         pause(0.1);
%     end
% end
% return;

% stores the summed input activity for recurrent connections in
% intermediate layer
rkl             = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);

% dynamics of the relaxation in the intermediate layer

% current activity in the intermediate layer
rij             = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);
rij_ant         = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);

% integrated activity (absolute value) after convergence
rij_final       = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);
rij_final_ant   = zeros(POPULATION_SIZE+1, POPULATION_SIZE+1);

% integration step 
h = 1;

% run the network
for t = 1:CONVERGENCE_STEPS
    
% loop through the projection layer
for idx=1:POPULATION_SIZE+1
    for jdx=1:POPULATION_SIZE+1       
        
        % recurrent connectivity contribution to overall activity
        for k = 1:POPULATION_SIZE+1
            for l = 1:POPULATION_SIZE+1
                if(k~=idx && l~=jdx)
                    rkl(idx, jdx) = rkl(idx, jdx) + W(idx,jdx,k,l)*...
                                                    rij_ant(k,l);
                end
            end
        end
        
        % superimpose contributions from both populations and reccurency
        rij(idx,jdx) = rxy_normalized(idx,jdx) + rkl(idx,jdx);
    end
end
        % normalize the total activity
        rij = normalize_activity(rij,...
                                 POPULATION_MIN_RATE, ...
                                 POPULATION_MAX_RATE);

for idx=1:POPULATION_SIZE+1
    for jdx=1:POPULATION_SIZE+1       
        % scaling factor of the activity when computing sigmoid
        scale_f  = 0.05;
        % threshold rate for the sigmoid
        threshold_rate = 75;  % spk/s
        % compute the activation for each neuron - sigmoid activation
        rij(idx,jdx) = sigmoid(POPULATION_MAX_RATE, ...
                               threshold_rate, ...
                               scale_f,...
                               rij(idx,jdx));
         
        % integrate activity in time (t->Inf) (External Euler method)
        % rij(t) = rij(t-1) + h*(drij(t)/dt)
        %           or in our case
        % rij(t) = rij(t-1) + h*(sigmoid(summed_activity(inputs, recurrency)) - rij(t-1))
        rij_final(idx,jdx) = rij_ant(idx, jdx) + h*((rij(idx,jdx) - rij_ant(idx, jdx)));
                                 
        % update history 
        rij_ant(idx, jdx) = rij_final(idx, jdx);
 
    end 
end
        % normalize the final activity
        rij_final = normalize_activity(rij_final,...
                                       POPULATION_MIN_RATE, ...
                                       POPULATION_MAX_RATE);
                                   
end % convergence steps

%% FEEDFORWARD CONNECTIVITY FROM INTERMEDIATE LAYER TO OUTPUT POPULATION
% after relaxation the intermediate layer activity is projected onto the
% output population after relaxation (t->Inf)

% sum of activity from intermediate layer to output layer
sum_interm_out = 0; 
for idx=1:POPULATION_SIZE+1
    sum_interm_out = 0;
    for jdx= 1:POPULATION_SIZE+1
        for k = 1:POPULATION_SIZE+1
            sum_interm_out = sum_interm_out + ...
                             G_map(0.05, z_population(idx).vi - phi(x_population(jdx).vi, y_population(k).vi))*...
                             (rij_final(jdx, k));
        end
    end
    z_population(idx).ri = sum_interm_out;
end

% normalize the activity in the output population
out = [z_population.ri];
out = normalize_activity(out,...
                         POPULATION_MIN_RATE,...
                         POPULATION_MAX_RATE);
for idx=1:POPULATION_SIZE+1
    z_population(idx).ri = out(idx);
end
                     
%% VISUALIZATION OF INTERMEDIATE LAYER AND OUTPUT POPULATION ACTIVITIES (AFTER DYNAMICS)
% intermediate layer activity after net dynamics relaxed
h2 = subplot(6, 4, [11 15]);
surf(rij_final(1:POPULATION_SIZE, 1:POPULATION_SIZE));
grid off;
set(gca, 'Box', 'off');  
% link axes for the 2 presentations of the intermediate layer activity
linkprop([h1 h2], 'CameraPosition');

% plot the encoded value in the output population after the network relaxed
% and the values are settles
subplot(6, 4, [23 24]);
z_population = init_population(z_population, POPULATION_SIZE+1, 'inferred_output');