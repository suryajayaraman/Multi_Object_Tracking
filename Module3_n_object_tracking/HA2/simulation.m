% This script is provided to you. You can use it to generate your own tracking scenario 
% and apply the n-object trackers you have implemented. Try to compare the estimation 
% performance of different trackers, from simple scenario with linear motion/measurement 
% model, high object detection probability and low clutter rate to more complex scenario 
% with nonlinear motion/measurement model, low detection probability and high clutter 
% rate. In complex scenarios, it should be easy to verify that, on average, the n-object
% tracker using multiple hypothesis solutions has the best performance. You can also 
% generate your own groundtruth data. How do the different trackers behave when multiple 
% objects move in close proximity? What do you observe? We also encourage you to tune the 
% different model parameters and observe how they affact the tracking performance. You 
% might also be interested in writting your own plotting function to illustrate the 
% estimation uncertainty of the object state and study how it changes over time. 

clear; close all; clc
dbstop if error

%Choose object detection probability
P_D = 0.9;
%Choose clutter rate
lambda_c = 10;

%Choose linear or nonlinear scenario
scenario_type = 'linear';

%Create tracking scenario
switch(scenario_type)
    case 'linear'
        %Creat sensor model
        range_c = [-1000 1000;-1000 1000];
        sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);
        
        %Creat linear motion model
        T = 1;
        sigma_q = 5;
        motion_model = motionmodel.cvmodel(T,sigma_q);
        
        %Create linear measurement model
        sigma_r = 10;
        meas_model = measmodel.cvmeasmodel(sigma_r);
        
        %Creat ground truth model
        nbirths = 5;
        K = 100;
        tbirth = zeros(nbirths,1);
        tdeath = zeros(nbirths,1);
        
        initial_state = repmat(struct('x',[],'P',eye(motion_model.d)),[1,nbirths]);
        
        initial_state(1).x = [0; 0; 0; -10];        tbirth(1) = 1;   tdeath(1) = K;
        initial_state(2).x = [400; -600; -10; 5];   tbirth(2) = 1;   tdeath(2) = K;
        initial_state(3).x = [-800; -200; 20; -5];  tbirth(3) = 1;   tdeath(3) = K;
        initial_state(4).x = [0; 0; 7.5; -5];       tbirth(4) = 1;   tdeath(4) = K;
        initial_state(5).x = [-200; 800; -3; -15];  tbirth(5) = 1;   tdeath(5) = K;
          
    case 'nonlinear'
        %Create sensor model
        %Range/bearing measurement range
        range_c = [-1000 1000;-pi pi];
        sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);
        
        %Create nonlinear motion model (coordinate turn)
        T = 1;
        sigmaV = 1;
        sigmaOmega = pi/180;
        motion_model = motionmodel.ctmodel(T,sigmaV,sigmaOmega);
        
        %Create nonlinear measurement model (range/bearing)
        sigma_r = 5;
        sigma_b = pi/180;
        s = [300;400];
        meas_model = measmodel.rangebearingmeasmodel(sigma_r, sigma_b, s);
        
        %Creat ground truth model
        nbirths = 4;
        K = 100;
        
        initial_state = repmat(struct('x',[],'P',diag([1 1 1 1*pi/90 1*pi/90].^2)),[1,nbirths]);
        
        initial_state(1).x = [0; 0; 5; 0; pi/180];       tbirth(1) = 1;   tdeath(1) = K;
        initial_state(2).x = [20; 20; -20; 0; pi/90];    tbirth(2) = 1;   tdeath(2) = K;
        initial_state(3).x = [-20; 10; -10; 0; pi/360];  tbirth(3) = 1;   tdeath(3) = K;
        initial_state(4).x = [-10; -10; 8; 0; pi/270];   tbirth(4) = 1;   tdeath(4) = K;
end

%Generate true object data (noisy or noiseless) and measurement data
ground_truth = modelgen.groundtruth(nbirths,[initial_state.x],tbirth,tdeath,K);
ifnoisy = 0;
objectdata = objectdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(objectdata,sensor_model,meas_model);

%N-object tracker parameter setting
P_G = 0.999;            %gating size in percentage
w_min = 1e-3;           %hypothesis pruning threshold
merging_threshold = 2;  %hypothesis merging threshold
M = 100;                %maximum number of hypotheses kept in MHT
density_class_handle = feval(@GaussianDensity);    %density class handle
tracker = n_objectracker();
tracker = tracker.initialize(density_class_handle,P_G,meas_model.d,w_min,merging_threshold,M);


%% copy variables for testing code
obj = tracker;
states = initial_state;
Z = measdata;
sensormodel = sensor_model;
motionmodel = motion_model;
measmodel = meas_model;


%% useful variables declaration
totalTrackTime = numel(Z);
n_objects = size(states,2);
measurement_size = size(Z{1},1);

% placeholders for outputs
estimates = cell(totalTrackTime, 1);

% useful parameters
log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
log_wk_zero_factor  = log(1 - sensormodel.P_D);
M = obj.reduction.M;

local_h_trees = cell(1, n_objects);
global_H = ones(1, n_objects);
w_log_h = log(1);

%% start of iteration loop
for i = 1 : n_objects
    local_h_trees{i} = states(i);
end

% for k = 1 : totalTrackTime
k = 1;

% get current timestep measurements
zk = Z{k};

% initialise cost matrix for solving optimisation
mk = size(zk, 2);


%% LOCAL HYPOTHESIS ELLIPSOIDAL GATING
% z_masks is a cell array (~dict in python) of length = n_objects.
% For each cell (meaning object), we store the local hypothesis for the
% object as a list of ```state``` class where the length of the list 
% is equal to the number of valid local hypothesis for the object
z_masks = cell(1, n_objects);

for i = 1 : n_objects
    lhs = local_h_trees{i};
    n_lhs = length(lhs);
    z_masks{i} = zeros(mk, n_lhs);
    
    for i_lhs = 1 : n_lhs
        [~, z_masks{i}(:, i_lhs)] = obj.density.ellipsoidalGating(lhs(i_lhs), zk, measmodel, obj.gating.size);
    end
end

% eliminate measurements not associated to any local hypotheses
disp(cell2mat(z_masks));
non_associated_z_indices = sum((cell2mat(z_masks)')) == 0;
zk = zk(:, ~non_associated_z_indices);

% reset number of valid measurements
z_masks = cellfun(@(z_masks_i) z_masks_i(~non_associated_z_indices,:), ...
                    z_masks, 'UniformOutput' ,false);
mk = sum(~non_associated_z_indices);
fprintf('After eliminating unassociated measurements, z_masks =\n');
disp(cell2mat(z_masks));


%% LOCAL HYPOTHESIS UPDATE STEP
% 2). calculate missed detection and predicted likelihood for
%     each measurement inside the gate and make sure to save
%     these for future use; 
% 3). create updated local hypotheses and make sure to save how
%     these connects to the old hypotheses and to the new the 
%     measurements for future use;

new_local_h_trees = cell(1, n_objects);
w_log_states = cell(1, n_objects);

% need to perform GSF for each object (local hypothesis)
% independently and store the weights and updated states
for i = 1 : n_objects
    lhs = local_h_trees{i};
    n_lhs = length(lhs);
    
    % EXAMPLE : suppose there is 2 local hypothesis for the 1st object and 5
    % measurements; size(w_log_states{1}) = (1, 12) and the structure is as
    % follows : [lh1 <-> 5 measurements, lh1_misdetection, ...
    %            lh2 <-> 5 measurements, lh2_misdetection]
    w_log_states{i} = -inf(1, n_lhs * (mk+1));

    for lh_i = 1 : n_lhs
        lh = lhs(lh_i);
        S_lh_i = tracker.computeInnovationCovariance(lh, measmodel);
        lhi_log = log_wk_theta_factor - 0.5 * log(det(2 * pi * S_lh_i));
        zhat_i = measmodel.h(lh.x);
        
        lh_i_gatedMeasurements_index = find(z_masks{i}(:, lh_i));
        for j = lh_i_gatedMeasurements_index
            new_lh_i = (lh_i -1) * (mk + 1) + j;
            w_log_states{i}(new_lh_i) = -(lhi_log - 0.5 * (zk(:,j) - zhat_i)' / S_lh_i * (zk(:,j) - zhat_i));
            new_local_h_trees{i}(new_lh_i) = obj.density.update(lh, zk(:,j), measmodel);
        end
        
       lh_i_misdetect_index = lh_i * (mk + 1);
       new_local_h_trees{i}(:, lh_i_misdetect_index) = lh;
       w_log_states{i}(lh_i_misdetect_index) = log_wk_zero_factor;
    end
end

    

%% GLOBAL HYPOTHESES COST MATRIX
% for each predicted global hypothesis: 
% 1). create 2D cost matrix; 

new_global_h = [];
new_global_w_h = [];

% in global hypotheses, each row reporsents a global hypothesis
% size = (n_global_hypotheses, n_objects)
% iterating through each of past global hypotheses
for i = 1 : size(global_H, 1)
end

    



            % 2). obtain M best assignments using a provided M-best 2D 
            %     assignment solver; 
            % 3). update global hypothesis look-up table according to 
            %     the M best assignment matrices obtained and use your new 
            %     local hypotheses indexing;


%% results plotting

%GNN filter
% GNNestimates = GNNfilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
% GNN_RMSE = RMSE_n_objects(objectdata.X,GNNestimates);
% X = sprintf('Root mean square error: GNN: %.3f',GNN_RMSE);
% disp(X)
% 
% 
% %JPDA filter
% JPDAestimates = JPDAfilter(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
% JPDA_RMSE = RMSE_n_objects(objectdata.X,JPDAestimates);
% fprintf('Root mean square error: JPDA: %.3f \n',JPDA_RMSE);
% 
% 
% 
% % %Multi-hypothesis tracker
% % TOMHTestimates = TOMHT(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);
% % TOMHT_RMSE = RMSE_n_objects(objectdata.X,TOMHTestimates);
% 
% 
% %Ploting
% figure
% hold on
% grid on
% 
% for i = 1:nbirths
%     h1 = plot(cell2mat(cellfun(@(x) x(1,i), objectdata.X, 'UniformOutput', false)), ...
%         cell2mat(cellfun(@(x) x(2,i), objectdata.X, 'UniformOutput', false)), 'g', 'Linewidth', 2);
%     h2 = plot(cell2mat(cellfun(@(x) x(1,i), GNNestimates, 'UniformOutput', false)), ...
%         cell2mat(cellfun(@(x) x(2,i), GNNestimates, 'UniformOutput', false)), 'r-s', 'Linewidth', 1);
%     h3 = plot(cell2mat(cellfun(@(x) x(1,i), JPDAestimates, 'UniformOutput', false)), ...
%         cell2mat(cellfun(@(x) x(2,i), JPDAestimates, 'UniformOutput', false)), 'm-o', 'Linewidth', 1);
% %     h4 = plot(cell2mat(cellfun(@(x) x(1,i), TOMHTestimates, 'UniformOutput', false)), ...
% %         cell2mat(cellfun(@(x) x(2,i), TOMHTestimates, 'UniformOutput', false)), 'b-d', 'Linewidth', 1);
% end
% 
% xlabel('x'); ylabel('y')
% % legend([h1 h2 h3 h4],'Ground Truth','GNN','JPDA','TOMHT', 'Location', 'best')
% legend([h1 h2 h3],'Ground Truth','GNN','JPDA', 'Location', 'best')
% set(gca,'FontSize',12) 
