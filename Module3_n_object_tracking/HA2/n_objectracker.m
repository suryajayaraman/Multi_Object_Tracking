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
        %INITIATOR initializes n_objectrackersei se ela acabou operando hoje ou nao, ou sei la o que class
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

        % STEPS FOR GNN
        % 0. perform prediction of each prior
        % 1. implement ellipsoidal gating for each predicted local hypothesis seperately, see Note below for details;
        % 2. construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
        % 3. find the best assignment matrix using a 2D assignment solver;
        % 4. create new local hypotheses according to the best assignment matrix obtained;
        % 5. extract object state estimates;
        % 6. predict each local hypothesis.
		
        % number of time steps
        total_track_time = length(Z);  

        estimates_x = cell(total_track_time,1);
        estimates_P = cell(total_track_time,1);

        % no of objects
        no_of_objects = numel(states);

        w_theta_k_factor = log(sensormodel.P_D / sensormodel.intensity_c);
        w_theta_0_factor = log(1 - sensormodel.P_D);

        for i = 1 :total_track_time
            z = Z{i};

            meas_in_gate = zeros(no_of_objects, size(z,2));

            for i = 1 : no_of_objects
                [~, meas_in_obj_gate] = obj.density.ellipsoidalGating(states(i), z, measmodel, obj.gating.size);
                meas_in_gate(i, :) = meas_in_obj_gate;
            end

            valid_meas_idx = sum(meas_in_gate, 1) > 0;
            meas_in_gate = meas_in_gate(:, valid_meas_idx);
            z_ingate = z(:, valid_meas_idx);
            mk = size(z_ingate, 2);

            states_upd = states;

            if(mk > 0)
                % calculate cost matrix
                cost_matrix = zeros(no_of_objects, no_of_objects + mk);
                % misdetection factors for all objects is same
                cost_matrix(:, mk +1 :end) = eye(no_of_objects) * w_theta_0_factor;

                % detection cost added
                for j = 1 : no_of_objects
                    if sum(meas_in_gate(j,:)) > 0
                        disp(j);
                        for k = 1 : mk
                            if meas_in_gate(j,k) == 1
                                cost_matrix(j,k) = w_theta_k_factor + obj.density.predictedLikelihood(states(j),z_ingate(:, k),measmodel);
                            end
                        end
                    end
                end

                % find the best assignment matrix for the calculated cost matrix
                [col4row, ~, gain] = assign2D(cost_matrix);

                % infeasible solution
                if(gain ~= -1)
                    %update detected objects according to best hypothesis
                    for row = 1: no_of_objects
                        if col4row(row) <= mk
                            z_obj_NN = z_ingate(:,col4row(row));    % nearest neighbour measurement
                            states_upd(row) = obj.density.update(states(row), z_obj_NN, measmodel);                
                        end      
                    end
                end 
            end

            % updated stat 
            estimates_x{i} = cell(arrayfun(@(state) (state.x), states_upd, 'UniformOutput',false));
            estimates_P{i} = cell(arrayfun(@(state) (state.P), states_upd, 'UniformOutput',false));

            % predict the next state
            states = arrayfun(@(state) obj.density.predict(state, motionmodel), states_upd);
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

        % STEPS FOR JPDA
        % 1. implement ellipsoidal gating for each local hypothesis seperately;
        % 2. construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
        % 3. find the M best assignment matrices using a M-best 2D assignment solver;
        % 4. normalise the weights of different data association hypotheses;
        % 5. prune assignment matrices that correspond to data association hypotheses with low weights and renormalise the weights;
        % 6. create new local hypotheses for each of the data association results;
        % 7. merge local hypotheses that correspond to the same object by moment matching;
        % 8. extract object state estimates;
        % 9. predict each local hypothesis.

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

        
        % STEPS FOR TOMHT
        % 1. for each local hypothesis in each hypothesis tree:
            % 1.1. implement ellipsoidal gating;
        % 2. disconsider measurements that do not fall inside any local hypothesis gate
        % 3. for each local hypothesis in each hypothesis tree:
            % 3.1. calculate missed detection and predicted likelihood for each measurement inside the gate and make sure to save these for future use; 
            % 3.2. create updated local hypotheses and make sure to save how these connects to the old hypotheses and to the new the measurements for future use;
        % 4. for each predicted global hypothesis: 
            % 4.1. create 2D cost matrix; 
            % 4.2. obtain M best assignments using a provided M-best 2D assignment solver; 
            % 4.3. update global hypothesis look-up table according to the M best assignment matrices obtained and use your new local hypotheses indexing;
        % 5. normalise global hypothesis weights and implement hypothesis reduction technique: pruning and capping;
        % 6. prune local hypotheses that are not included in any of the global hypotheses;
        % 7. Re-index global hypothesis look-up table;
        % 8. extract object state estimates from the global hypothesis with the highest weight;
        % 9. predict each local hypothesis in each hypothesis tree.

    end
    
end
end
