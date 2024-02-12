%% SIMPLE IMPLEMENTATION OF THE NSUPERVISED LEARNING OF RELATIONS NETWORK
% the demo dataset contains the y = f(x) relation
%% PREPARE ENVIRONMENT
clear all; clc; close all;
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL      = 1;
% number of populations in the network
N_POP           = 2;
% number of neurons in each population
N_NEURONS       = 200;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE  = 1;
% WTA circuit settling threshold
EPSILON         = 1e-3;
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest
MIN_VAL         = -1.0;
MAX_VAL         = 1.0;

traffic_demo = 1; % 0/1 with or without traffic data
traffic_pair = 1; % pair 1, 2, or 3 from the 3 vars NO2, Humidity, Vehicle Count
img = 'setup.png';

contains = @(str, pattern) ~cellfun('isempty', strfind(str, pattern));

if traffic_demo == 1
    % load raw data
    %% Extract real-world data from Newcastle Urban Observatory (http://uoweb1.ncl.ac.uk/api_page/)
    % load data for each of the following scenarios:
    load('NUO_traffic_sample_aggregated.mat'); % for traffic scenario


    % follow an interpolation for data augmentation
    upsample_factor = 8;
    vehicle_count = [18, 22, 21, 26, 17, 12, 7, 15, 8, 8, 8, 10, 6, 7, 6, 7, 7, 2, 7, 6, 7, 3, 5, 3, 4, 3, 4, 3, 2];
    datax = vehicle_count;
    idx_data = 1:length(datax);
    idx_upsampled_data = 1:1/upsample_factor:length(datax);
    vehicle_count = interp1(idx_data, datax, idx_upsampled_data, 'linear');

    sensory_data.range  = 1.0;
    % convert to [-1, +1] range
    minVal = min(vehicle_count);
    maxVal = max(vehicle_count);
    vehicle_count = (((vehicle_count - minVal) * (sensory_data.range - (-sensory_data.range))) / (maxVal - minVal)) + (-sensory_data.range);

    [~, sensors_num] = size(site1_sensors);
    site1_sensors_extrapolated = [];
    for id = 1:sensors_num
        datax = site1_sensors(:,id)';
        idx_data = 1:length(datax);
        idx_upsampled_data = 1:1/upsample_factor:length(datax);
        site1_sensors_extrapolated(:,id) = interp1(idx_data, datax, idx_upsampled_data, 'linear');
    end

    [~, sensors_num] = size(site2_sensors);
    site2_sensors_extrapolated = [];
    for id = 1:sensors_num
        datax = site2_sensors(:,id)';
        idx_data = 1:length(datax);
        idx_upsampled_data = 1:1/upsample_factor:length(datax);
        site2_sensors_extrapolated(:,id) = interp1(idx_data, datax, idx_upsampled_data, 'linear');
    end

    site1_sensors = site1_sensors_extrapolated;
    site2_sensors = site2_sensors_extrapolated;

    [N_SAMPLES, sensors_num] = size(site1_sensors);
    % set up the interval of interest (i.e. +/- range)ststr
    uon_sensory_data.range  = 1.0;
    % convert to [-1, +1] range
    for id = 1:sensors_num
        minVal = min(site1_sensors(:, id));
        maxVal = max(site1_sensors(:, id));
        site1_sensors(:,id) = (((site1_sensors(:,id) - minVal) * (uon_sensory_data.range - (-uon_sensory_data.range))) / (maxVal - minVal)) + (-uon_sensory_data.range);
    end
    [~, sensors_num] = size(site2_sensors);
    for id = 1:sensors_num
        minVal = min(site2_sensors(:, id));
        maxVal = max(site2_sensors(:, id));
        site2_sensors(:,id) = (((site2_sensors(:,id) - minVal) * (uon_sensory_data.range - (-uon_sensory_data.range))) / (maxVal - minVal)) + (-uon_sensory_data.range);
    end
    % setup the number of random input samples to generate
    uon_sensory_data.num_vals =  N_SAMPLES;
    switch traffic_pair
        case 1
            % (a) Learn within location between sensor correlations
            % example NO2 and Humidity for site 1
            uon_sensory_data.x = site1_sensors(1:uon_sensory_data.num_vals, contains(site1_sensors_names,'NO2'));
            uon_sensory_data.y = site2_sensors(1:uon_sensory_data.num_vals, contains(site2_sensors_names,'Humidity'));
            img = 'setup1.png';
        case 2
            % example NO2 and Vehicle Count for site 1
            uon_sensory_data.x = site1_sensors(1:uon_sensory_data.num_vals, contains(site1_sensors_names,'NO2'));
            uon_sensory_data.y = vehicle_count;
            img = 'setup2.png';
        case 3
            % example Humidity and Vehicle Count for site 1
            uon_sensory_data.x = site2_sensors(1:uon_sensory_data.num_vals, contains(site2_sensors_names,'Humidity'));
            uon_sensory_data.y = vehicle_count;
            img = 'setup3.png';
    end
    % data is scaled, compressed top [-1, 1] for neural computation
    sensory_data.x  = MIN_VAL + rand(N_SAMPLES, 1)*(MAX_VAL - MIN_VAL);
    load('relation_fits_traffic.mat');
    switch traffic_pair
        case 1
            % example NO2 and Humidity for site 1
            sensory_data.y = polyval(fit_exp0.coeff, sensory_data.x);
        case 2
            % example NO2 and Vehicle Count for site 1
            sensory_data.y = polyval(fit_exp1.coeff, sensory_data.x);
        case 3
            % example Humidity and Vehicle Count for site 1
            sensory_data.y = polyval(fit_exp2.coeff, sensory_data.x);
    end
else
    % setup the number of random input samples to generate
    NUM_VALS        = 250;
    % generate NUM_VALS random samples in the given interval
    sensory_data.x  = MIN_VAL + rand(NUM_VALS, 1)*(MAX_VAL - MIN_VAL);
    sensory_data.y  = (power(sensory_data.x, 2));
end
DATASET_LEN     = length(sensory_data.x);
%% INIT NETWORK DYNAMICS
% epoch iterator in outer loop (HL, HAR)
t       = 1;
% network iterator in inner loop (WTA)
tau     = 1;
% constants for WTA circuit (convolution based WTA), these will provide a
% profile peaked at ~ TARGET_VAL_ACT
DELTA   = -0.005;                   % displacement of the convolutional kernel (neighborhood)
SIGMA   = 5.0;                      % standard deviation in the exponential update rule
SL      = 4.5;                      % scaling factor of neighborhood kernel
GAMMA   = SL/(SIGMA*sqrt(2*pi));    % convolution scaling factor
% constants for Hebbian linkage
ALPHA_L = 1.0*1e-2;                 % Hebbian learning rate
ALPHA_D = 1.0*1e-2;                 % Hebbian decay factor ALPHA_D >> ALPHA_L
% constants for HAR
C       = 0.005;                    % scaling factor in homeostatic activity regulation
TARGET_VAL_ACT  = 0.4;              % amplitude target for HAR
A_TARGET        = TARGET_VAL_ACT*ones(N_NEURONS, 1); % HAR target activity vector
% constants for neural units in neural populations
M       = 1; % slope in logistic function @ neuron level
S       = 10.0; % shift in logistic function @ neuron level
% activity change weight (history vs. incoming knowledge)
ETA     = 0.25;
%% CREATE NETWORK AND INITIALIZE
% create a network given the simulation constants
populations = create_init_network(N_POP, N_NEURONS, GAMMA, SIGMA, DELTA, MAX_INIT_RANGE, TARGET_VAL_ACT);
% buffers for changes in activity in WTA loop
act = zeros(N_NEURONS, N_POP)*MAX_INIT_RANGE;
old_act = zeros(N_NEURONS, N_POP)*MAX_INIT_RANGE;
% buffers for running average of population activities in HAR loop
old_avg = zeros(N_POP, N_NEURONS);
cur_avg = zeros(N_POP, N_NEURONS);
% the new rate values
delta_a1 = zeros(N_NEURONS, 1);
delta_a2 = zeros(N_NEURONS, 1);
%% NETWORK SIMULATION LOOP
% % present each entry in the dataset for MAX_EPOCHS epochs to train the net
for didx = 1:DATASET_LEN
    % pick a new sample from the dataset and feed it to the input (noiseless input)
    % population in the network (in this case X -> A -> | <- B <- Y)
    X = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
    Y = population_encoder(sensory_data.y(didx), max(sensory_data.y(:)),  N_NEURONS);
    % normalize input such that the activity in all units sums to 1.0
    X = X./sum(X);
    Y = Y./sum(Y);
    % clamp input to neural populations
    populations(1).a = X;
    populations(2).a = Y;
    % given the input sample wait for WTA circuit to settle and then
    % perform a learning step of Hebbian learning and HAR
    while(1)
        % compute changes in activity
        delta_a1 = compute_s(populations(1).h + populations(1).Wext*populations(2).a + populations(1).Wint*populations(1).a, M, S);
        delta_a2 = compute_s(populations(2).h + populations(2).Wext*populations(1).a + populations(2).Wint*populations(2).a, M, S);
        % update the activities of each population
        populations(1).a = (1-ETA)*populations(1).a + ETA*delta_a1;
        populations(2).a = (1-ETA)*populations(2).a + ETA*delta_a2;
        % current activation values holder
        for pop_idx = 1:N_POP
            act(:, pop_idx) = populations(pop_idx).a;
        end
        % check if activity has settled in the WTA loop
        q = (sum(sum(abs(act - old_act)))/(N_POP*N_NEURONS));
        if(q <= EPSILON)
            tau = 1;
            break;
        end
        % update history of activities
        old_act = act;
        % increment time step in WTA loop
        tau = tau + 1;
        % visualize runtime data
        if(DYN_VISUAL==1)
            if traffic_demo == 1
                load('relation_fits_traffic.mat');
                switch traffic_pair
                    case 1
                        % example NO2 and Humidity for site 1
                        visualize_runtime_traffic_setup(uon_sensory_data, sensory_data, populations, didx, 'NO2', 'Humidity', img);
                    case 2
                        % example NO2 and Vehicle Count for site 1
                        visualize_runtime_traffic_setup(uon_sensory_data,sensory_data, populations, didx, 'NO2', 'VehicleCount', img);
                    case 3
                        % example Humidity and Vehicle Count for site 1
                        visualize_runtime_traffic_setup(uon_sensory_data,sensory_data, populations, didx, 'Humidity','VehicleCount', img);
                end
            else
                % visualize post-simulation data
                visualize_runtime(sensory_data, populations, 1, t, didx);
            end
        end
    end  % WTA convergence loop
    % update Hebbian linkage between the populations (decaying Hebbian rule)
    populations(1).Wext = (1-ALPHA_D)*populations(1).Wext + ALPHA_L*populations(1).a*populations(2).a';
    populations(2).Wext = (1-ALPHA_D)*populations(2).Wext + ALPHA_L*populations(2).a*populations(1).a';
    % compute the inverse time for exponential averaging of HAR activity
    omegat = 0.002 + 0.998/(t+2);
    % for each population in the network
    for pop_idx = 1:N_POP
        % update Homeostatic Activity Regulation terms
        % compute exponential average of each population at current step
        cur_avg(pop_idx, :) = (1-omegat)*old_avg(pop_idx, :) + omegat*populations(pop_idx).a';
        % update homeostatic activity terms given current and target act.
        populations(pop_idx).h = populations(pop_idx).h + C*(TARGET_VAL_ACT - cur_avg(pop_idx, :)');
    end
    % update averging history
    old_avg = cur_avg;
    % increment timestep for HL and HAR loop
    t = t + 1;
end % end of all samples in the training dataset

% for the traffic go scenario use different visualizers for each case.
if traffic_demo == 1
    load('relation_fits_traffic.mat');
    switch traffic_pair
        case 1
            % example NO2 and Humidity for site 1
            visualize_runtime_traffic_setup(uon_sensory_data, sensory_data, populations, didx, 'NO2', 'Humidity', img);
        case 2
            % example NO2 and Vehicle Count for site 1
            visualize_runtime_traffic_setup(uon_sensory_data, sensory_data, populations, didx, 'NO2', 'VehicleCount', img);
        case 3
            % example Humidity and Vehicle Count for site 1
            visualize_runtime_traffic_setup(uon_sensory_data, sensory_data, populations, didx, 'Humidity','VehicleCount', img);
    end
else
    % visualize post-simulation data
    visualize_runtime(sensory_data, populations, 1, t, DATASET_LEN);
end
