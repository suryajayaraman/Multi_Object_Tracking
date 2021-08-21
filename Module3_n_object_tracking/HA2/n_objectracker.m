classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in
    %clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes n_objectracker class
            %INPUT: density_class_handle: density class handle
            %       P_D: object detection probability
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         used in TOMHT --- scalar 
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function S = computeInnovationCovariance(obj, state, measmodel)
            %Measurement model Jacobian
            Hx = measmodel.H(state.x);
            %Innovation covariance
            S = Hx * state.P * Hx' + measmodel.R;
            %Make sure matrix S is positive definite
            S = (S+S') / 2;     
        end
       
        

        function [estimates_x, estimates_P] = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %GNNFILTER tracks n object using global nearest neighbor
            %association 
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)  

            totalTrackTime = numel(Z);
            n_objects = size(states,2);
            state_size = size(states(1).x,1);

            % placeholders for outputs
            estimates   = cell(totalTrackTime, 1);
            estimates_x = cell(totalTrackTime, 1);
            estimates_P = cell(totalTrackTime, 1);

            % useful parameters
            log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_wk_zero_factor  = log(1 - sensormodel.P_D);
            
            for k = 1 : totalTrackTime
                % get current timestep measurements
                zk = Z{k};

                % initialise cost matrix for solving optimisation
                mk = size(zk, 2);
                costMatrix = inf(n_objects, mk + n_objects);

                for n = 1 : n_objects
                    state = states(n);
					Shat_i = obj.computeInnovationCovariance(state, measmodel);
					li_log = log_wk_theta_factor - 0.5 * log(det(2 * pi * Shat_i));
                    zhat_i = measmodel.h(state.x);
                    
                    % gate for each object
                    [~, zk_inGateIndices] = obj.density.ellipsoidalGating(state, zk, measmodel, obj.gating.size);
                %     disp(zk_inGateIndices');
                    for zi = 1 : mk
                        if zk_inGateIndices(zi) == true
                            costMatrix(n, zi) = -(li_log - 0.5 * (zk(:,zi) - zhat_i)' / Shat_i * (zk(:,zi) - zhat_i));
                        end      
                    end

                    % misdetection cost
                    costMatrix(n, n+ mk) = log_wk_zero_factor;
                end

                % filter cols containing only inf values
                noninf_cols = sum(isinf(costMatrix)) < n_objects;
                costMatrix = costMatrix(:, noninf_cols);
                zk_valid = zk(:, noninf_cols(1:mk));
                mk_valid = size(zk_valid, 2);

                estimates{k}   = zeros(state_size, n_objects);
                estimates_x{k} = zeros(state_size, n_objects);
                estimates_P{k} =  ones(state_size, state_size, n_objects);
                
                % solve assignment problem
                [col4row, ~, gain]=assign2D(costMatrix);
                if(gain ~= -1)
                    for n = 1 : n_objects
                        state = states(1,n);

                        % perform kalman filter update for measurement associated 
                        % with object in current hypothesis
                        if col4row(n,1) <= mk_valid
                            z_NN_n = zk_valid(:, col4row(n,1));
                            state = obj.density.update(state, z_NN_n, measmodel);
                        end

                        % store posterior density
                        estimates{k}(:,n)     = state.x;
                        estimates_x{k}(:,n)   = state.x;
                        estimates_P{k}(:,:,n) = state.P;

                        % kalman filter prediction
                        state = obj.density.predict(state,motionmodel);
                        states(n) = state;
                    end
                end            
            end
        end
                
        
        function [estimates_x, estimates_P] = JPDAfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %JPDAFILTER tracks n object using joint probabilistic data
            %association
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            %--------------------------------------------------------------------------------
            % 1. implement ellipsoidal gating for each local hypothesis seperately;
            % 2. construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
            % 3. find the M best assignment matrices using a M-best 2D assignment solver;
            % 4. normalise the weights of different data association hypotheses;
            % 5. prune assignment matrices that correspond to data association hypotheses with low weights and renormalise the weights;
            % 6. create new local hypotheses for each of the data association results;
            % 7. merge local hypotheses that correspond to the same object by moment matching;
            % 8. extract object state estimates;
            % 9. predict each local hypothesis.
            %--------------------------------------------------------------------------------

            totalTrackTime = numel(Z);
            n_objects = size(states,2);
            measurement_size = size(Z{1},1);
            state_size = size(states(1).x,1);

            % placeholders for outputs
            estimates   = cell(totalTrackTime, 1);
            estimates_x = cell(totalTrackTime, 1);
            estimates_P = cell(totalTrackTime, 1);
            
            % useful parameters
            log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_wk_zero_factor  = log(1 - sensormodel.P_D);
            M = obj.reduction.M;

            
            for k = 1 : totalTrackTime
                % get current timestep measurements
                zk = Z{k};

                % initialise cost matrix for solving optimisation
                mk = size(zk, 2);

                % 2. construct 2D cost matrix of size (number of objects, number of measurements 
                % that at least fall inside the gates + number of objects);				
                costMatrix = inf(n_objects, mk + n_objects);

                for n = 1 : n_objects
                    state = states(n);
                    Shat_i = obj.computeInnovationCovariance(state, measmodel);
                    li_log = log_wk_theta_factor - 0.5 * log(det(2 * pi * Shat_i));
                    zhat_i = measmodel.h(state.x);

                    % 1. implement ellipsoidal gating for each local hypothesis seperately;
                    [~, zk_inGateIndices] = obj.density.ellipsoidalGating(state, zk, measmodel, obj.gating.size);
                    for zi = 1 : mk
                        if zk_inGateIndices(zi) == true
                            costMatrix(n, zi) = -(li_log - 0.5 * (zk(:,zi) - zhat_i)' / Shat_i * (zk(:,zi) - zhat_i));
                        end      
                    end

                    % misdetection cost
                    costMatrix(n, n+ mk) = -log_wk_zero_factor;
                end

                % filter cols containing only inf values
                noninf_cols = sum(isinf(costMatrix)) < n_objects;
                costMatrix = costMatrix(:, noninf_cols);
                zk_valid = zk(:, noninf_cols(1:mk));
                mk_valid = size(zk_valid,2);

                
                % 3. find the M best assignment matrices using a M-best 2D assignment solver;
                [col4rowBest, ~, ~]= kBest2DAssign(costMatrix, M);
                % assert(all(gainBest~=-1), 'not solvable');
                
                % number of valid hypothesis in output may not be as desired, so check again for 
                % number of valid hypothesis calculated
                M_left = size(col4rowBest, 2);

                % placeholder for storing log-weights of hypothesis
                hypothesis = 1 : M_left;                
                log_w_h = zeros(1, M_left);
                for i = 1 : M_left
                    tr_AL_h = sum(costMatrix(sub2ind(size(costMatrix), 1:n_objects, col4rowBest(:, i)')));
                    log_w_h(1, i) = -tr_AL_h;
                end                                

                % 4. normalise the weights of different data association hypotheses;
                log_w_h  = normalizeLogWeights(log_w_h);

                % consider all misdetections as same value
                col4rowBest(col4rowBest > mk_valid) = mk_valid + 1;

                
                % 5. prune assignment matrices that correspond to data association hypotheses 
                % with low weights and renormalise the weights;
                [log_w_h, hypothesis] = hypothesisReduction.prune(log_w_h, hypothesis, obj.reduction.w_min);
				col4rowBest = col4rowBest(:, hypothesis);
                log_w_h = normalizeLogWeights(log_w_h);
                
                % calculate Beta matrix
                betaMatrix = zeros(n_objects, mk_valid + 1);
                for objectIndex = 1 : n_objects
                    for hypothesisIndex = 1 : size(log_w_h,2)
                        measurementIndex = col4rowBest(objectIndex, hypothesisIndex);
                        betaMatrix(objectIndex, measurementIndex) = betaMatrix(objectIndex, measurementIndex) + ...
																	exp(log_w_h(hypothesisIndex));
                    end
                end

                estimates{k}   = zeros(state_size, n_objects);
                estimates_x{k} = zeros(state_size, n_objects);
                estimates_P{k} =  ones(state_size, state_size, n_objects);
                
                % 6. create new local hypotheses for each of the data association results;
                % 7. merge local hypotheses that correspond to the same object by moment matching;                
                for i = 1 : n_objects
                    state = states(i);
					Hx = measmodel.H(state.x);
                    S_i = obj.computeInnovationCovariance(state, measmodel);
                    K_i = state.P * Hx' / S_i;

                    eps = zeros(measurement_size, 1);
                    eps_ij_sq_sum = zeros(measurement_size, measurement_size);
                    
					for j = 1 : mk_valid
                        eps_ij = zk_valid(:,j) - measmodel.h(state.x);
                        eps = eps  + betaMatrix(i, j) * eps_ij;
                        eps_ij_sq_sum = eps_ij_sq_sum + (betaMatrix(i, j) * (eps_ij * eps_ij'));
                    end


                    % posterior state updates
                    state.x = state.x + K_i * eps;   

                    % temp variables for calculating posterior covariance
                    P_bar_i = state.P - K_i * S_i * K_i';
                    P_tilde_i = K_i * (eps_ij_sq_sum - eps  * eps') * K_i';
                    beta0 = betaMatrix(i, mk_valid + 1);					
                    state.P = beta0 * state.P + (1 - beta0) * P_bar_i + P_tilde_i;
                    
                    
                    % 8. extract object state estimates;
                    % store posterior density
                    estimates{k}(:,i)     = state.x;
                    estimates_x{k}(:,i)   = state.x;
                    estimates_P{k}(:,:,i) = state.P;                    

                    % 9. predict each local hypothesis.                    
                    % kalman filter prediction
                    state = obj.density.predict(state,motionmodel);
                    states(i) = state;
                end                
            end
        end
            
            
        function [estimates_x, estimates_P] = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
            %TOMHT tracks n object using track-oriented multi-hypothesis tracking
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            
            % Task description
            %
            % for each local hypothesis in each hypothesis tree: 
            % 1). implement ellipsoidal gating; 
            % 2). calculate missed detection and predicted likelihood for
            %     each measurement inside the gate and make sure to save
            %     these for future use; 
            % 3). create updated local hypotheses and make sure to save how
            %     these connects to the old hypotheses and to the new the 
            %     measurements for future use;
            %
            % for each predicted global hypothesis: 
            % 1). create 2D cost matrix; 
            % 2). obtain M best assignments using a provided M-best 2D 
            %     assignment solver; 
            % 3). update global hypothesis look-up table according to 
            %     the M best assignment matrices obtained and use your new 
            %     local hypotheses indexing;
            %
            % normalise global hypothesis weights and implement hypothesis
            % reduction technique: pruning and capping;
            %
            % prune local hypotheses that are not included in any of the
            % global hypotheses;
            %
            % Re-index global hypothesis look-up table;
            %
            % extract object state estimates from the global hypothesis
            % with the highest weight;
            %
            % predict each local hypothesis in each hypothesis tree.
            %
            %
            %% useful variables declaration
            totalTrackTime = numel(Z);
            n_objects = size(states,2);
            state_size = size(states(1).x,1);

            % placeholders for outputs
            estimates   = cell(totalTrackTime, 1);
            estimates_x = cell(totalTrackTime, 1);
            estimates_P = cell(totalTrackTime, 1);

            % useful parameters
            log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_wk_zero_factor  = log(1 - sensormodel.P_D);
            M = obj.reduction.M;

			% placeholders for Local hypotheses trees
            local_h_trees = cell(1, n_objects);
            for i = 1 : n_objects
                local_h_trees{i} = states(i);
            end
			
			% placeholders for global hypotheses trees
			global_h = ones(1, n_objects);
            w_log_h = log(1);
			
			% start of iteration loop
            for k = 1 : totalTrackTime
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
                % disp(cell2mat(z_masks));
                non_associated_z_indices = sum(cell2mat(z_masks)') == 0;
                zk = zk(:, ~non_associated_z_indices);

                % reset number of valid measurements
                z_masks = cellfun(@(z_masks_i) z_masks_i(~non_associated_z_indices,:), ...
                                    z_masks, 'UniformOutput' ,false);
                mk = sum(~non_associated_z_indices);
                % fprintf('After eliminating unassociated measurements, z_masks =\n');
                % disp(cell2mat(z_masks));


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
                        S_lh_i = obj.computeInnovationCovariance(lh, measmodel);
                        zhat_i = measmodel.h(lh.x);						
                        lhi_log = log_wk_theta_factor - 0.5 * log(det(2 * pi * S_lh_i));

                        lh_i_gatedMeasurements_index = find(z_masks{i}(:, lh_i))';
                        for j = lh_i_gatedMeasurements_index
                            new_lh_i = (lh_i -1) * (mk + 1) + j;
                            w_log_states{i}(new_lh_i) = lhi_log - 0.5 * (zk(:,j) - zhat_i)' / S_lh_i * (zk(:,j) - zhat_i);
                            new_local_h_trees{i}(new_lh_i) = obj.density.update(lh, zk(:,j), measmodel);
                        end

                       lh_i_misdetect_index = lh_i * (mk + 1);
                       new_local_h_trees{i}(:, lh_i_misdetect_index) = lh;
                       w_log_states{i}(lh_i_misdetect_index) = log_wk_zero_factor;
                    end
                end


                %% GLOBAL HYPOTHESES COST MATRIX
                % for each predicted global hypothesis, 1). create 2D cost matrix; 
                new_global_h = [];
                new_w_log_h = [];

                % in global hypotheses, each row represents a global hypothesis
                % size = (n_global_hypotheses, n_objects)
                % iterating through each of past global hypotheses
                for H_i = 1 : size(global_h, 1)

                    % cost matrix from prior global hypothesis
                    L_h = inf(n_objects, n_objects + mk);

                    % use the log weights computed for each local hypothesis to fill the
                    % cost matrix
                    for i = 1 : n_objects

                        % EXAMPLE : Consider there are 3 local hypotheses for nth object
                        % and there are 10 measurements at current time step
                        % [1  - 10][11] = lh_1 and lh_i = 1
                        % [12 - 21][22] = lh_2 and lh_i = 2
                        % [23 - 32][33] = lh_3 and lh_i = 3
                        lh_i = global_h(H_i, i);
                        lh_first_i = (lh_i - 1) * (mk + 1) + 1;
                        lh_last_i  = lh_i * (mk + 1) - 1;

                        % negative of the log likelihood weights and misdetection weights 
                        % from local hypotheses
                        L_h(i, 1 : mk) = -w_log_states{i}(lh_first_i : lh_last_i);
                        L_h(i, mk + i) = -w_log_states{i}(lh_last_i + 1);
                    end

                    % 3. find the M best assignment matrices using a M-best 2D assignment solver;
                    [col4rowBest, ~, ~]= kBest2DAssign(L_h, M);

                    % number of valid hypothesis in output may not be as desired, so check again for 
                    % number of valid hypothesis calculated
                    M_left = size(col4rowBest, 2);

                    % placeholder for storing log-weights of hypothesis
                    hypothesis = 1 : M_left;                

                    % computing the probability of the hyptheses as -trace(At * L_h)
                    new_w_log_h_i = arrayfun(@(h_i) -sum(L_h(sub2ind(size(L_h), ...
                                    1:n_objects, col4rowBest(:, h_i)'))), hypothesis);

                    % adding log weight at previous timestep
                    new_w_log_h_i = new_w_log_h_i + w_log_h(H_i);

                    % appending to new weights list
                    new_w_log_h = [new_w_log_h; new_w_log_h_i'];

                    % consider all misdetections as same value
                    col4rowBest(col4rowBest > mk) = mk + 1;

                    for M_i = 1 : M_left
                        new_global_h(end + 1, :) = arrayfun(@(i) ...
                            (global_h(H_i, i) - 1) * (mk + 1) + col4rowBest(i, M_i), 1:n_objects);
                    end    
                end


                %% PRUNE AND CAP GLOBAL HYPOTHESES
                % new_global_h is the look up table from local hypotheses where each column
                % refers to each object hypotheses. 
                % EXAMPLE : suppose a row in new_global_h is [1, 3, 5, 6, 4].
                % This means 1st object is assoicated with its 1st local hypotheses, 2nd
                % object with its own 3rd local hypotheses, 3rd object with its own 5th
                % hypotheses etc

                % normalise the weights of different data association hypotheses;
                new_w_log_h  = normalizeLogWeights(new_w_log_h);

                % prune assignment matrices that correspond to data association hypotheses 
                % with low weights and renormalise the weights;
                [new_w_log_h, hyp_left] = hypothesisReduction.prune(new_w_log_h, 1:length(new_w_log_h), obj.reduction.w_min);
                new_w_log_h = normalizeLogWeights(new_w_log_h);
                new_global_h = new_global_h(hyp_left, :);     

                % cap to maximum number of hypotheses and renormalise the weights;
                [new_w_log_h, hyp_left] = hypothesisReduction.cap(new_w_log_h, 1:length(new_w_log_h), M);
                new_w_log_h = normalizeLogWeights(new_w_log_h);
                new_global_h = new_global_h(hyp_left, :);     


                %% REMOVE UNUSED LOCAL HYPOTHESES
                % Of all the global hypotheses, for each object only certain local
                % hyptheses might be used; others can be pruned. This is done for each
                % object independently; EXAMPLE : there are 6 local hypotheses for 1st
                % object. But only [1,2,4] appear in all of global hypotheses together.
                % Hence, we can prune [3,5,6] from local hypotheses of 1st object

                for i = 1: n_objects
                    % select only those appearing in global hypotheses table
                    hyp_keep_i = unique(new_global_h(:,i));
                    new_local_h_trees{i} = new_local_h_trees{i}(hyp_keep_i);

                    % reindex the global hypothesis 
                    for new_index = 1 : numel(hyp_keep_i)
                       new_global_h(new_global_h(:,i) == hyp_keep_i(new_index), i) = new_index;
                    end
                end


                %% ESTIMATE MAX LIKELIHOOD HYPOTHESES
                [~, best_w_log_h_idx] = max(new_w_log_h);

                estimates{k}   = zeros(state_size, n_objects);
                estimates_x{k} = zeros(state_size, n_objects);
                estimates_P{k} =  ones(state_size, state_size, n_objects);

                for i = 1 : n_objects
                    % local hypotheses with highest weight is best estimate for object
                    local_hyp_index = new_global_h(best_w_log_h_idx, i);
                    estimates{k}(:, i)    = new_local_h_trees{i}(local_hyp_index).x;
                    estimates_x{k}(:, i)  = new_local_h_trees{i}(local_hyp_index).x;
                    estimates_P{k}(:,:,i) = new_local_h_trees{i}(local_hyp_index).P;

                    % prediction step for each local hypotheses
                    new_local_h_trees{i} = arrayfun( @(lh_i) obj.density.predict(...
                                        lh_i,motionmodel), new_local_h_trees{i});                    
                end

                %% RESET VARIABLES AT END OF TIMESTEP
                local_h_trees = new_local_h_trees;
                global_h = new_global_h;
                w_log_h = new_w_log_h;
            end
        end
    end
end
