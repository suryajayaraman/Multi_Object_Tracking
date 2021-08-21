classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
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
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
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
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function [estimates_x, estimates_P] = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1  
    	    
			totalTrackTime = size(Z,1);
                        
            % placeholders for outputs
            estimates = cell(totalTrackTime, 1);
            estimates_x = cell(totalTrackTime, 1);
            estimates_P = cell(totalTrackTime, 1);
	
            % useful parameters
            log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_wk_zero_factor  = log(1 - sensormodel.P_D);
	
            % initial state is considered posterior for first timestamp
            posteriorState = state;
	
            % iterate through timestamps
            for k = 1 : totalTrackTime
            
                % get current timestep measurements
                zk = Z{k};
                
                % perform gating and find number of measurements inside limits
                [z_inGate, ~] = obj.density.ellipsoidalGating(state, zk, measmodel, obj.gating.size);
                mk = size(z_inGate, 2) + 1;
                                
                % misdetection
                if mk == 1
                    posteriorState = state;
	
                % object is detected
                else
                    likelihoodDensity = obj.density.predictedLikelihood(state, z_inGate, measmodel); 
                    wk_theta = exp(log_wk_theta_factor + likelihoodDensity);
                    wk_zero  = exp(log_wk_zero_factor);

                    [max_wk_theta, index] = max(wk_theta);
                    if(max_wk_theta < wk_zero)
                        posteriorState = state;
                    else                    
                        % kalman filter update using nearest neighbour measurement
                        z_NN = z_inGate(:, index);
                        posteriorState = obj.density.update(state, z_NN, measmodel);
                    end
                end
                
                % updated state variables 
                estimates{k}   = posteriorState.x;
                estimates_x{k} = posteriorState.x;
                estimates_P{k} = posteriorState.P;
                
                % predict the next state
                state = obj.density.predict(posteriorState, motionmodel);
            end			
        end
        
        
        function [estimates_x, estimates_P] = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
            totalTrackTime = size(Z,1);
            
            % placeholders for outputs
            estimates = cell(totalTrackTime, 1);
            estimates_x = cell(totalTrackTime, 1);
            estimates_P = cell(totalTrackTime, 1);
            
            % useful parameters
            log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_wk_zero_factor  = log(1 - sensormodel.P_D);
            
            % initial state is considered posterior for first timestamp
            posteriorState = state;
            
            % iterate through timestamps
            for k = 1 : totalTrackTime
                % get current timestep measurements
                zk = Z{k};

                % perform gating and find number of measurements inside limits
                [z_inGate, ~] = obj.density.ellipsoidalGating(state, zk, measmodel, obj.gating.size);
                mk = size(z_inGate, 2) + 1;
                	
                % misdetection
                if mk == 1
                    posteriorState = state;
	
				% object is detected
                else		
                    % placeholder for storing possible hypothesis
                    hypothesisArray = repmat(state, mk,1);
                    % misdetection is the last state 
                    hypothesisArray(mk,1) = state;

                    % calculate predicted likelihood = N(z; zbar, S) => scalar value
                    % represents log weights for measurements inside gate
                    likelihoodDensity = obj.density.predictedLikelihood(state, z_inGate, measmodel);

                    % unnormalized weights vector
                    log_wk_theta = log_wk_theta_factor + likelihoodDensity;
                    log_wk_zero  = log_wk_zero_factor;
                    hypothesis_logWeights = [log_wk_theta; log_wk_zero];
                    % fprintf('\n Sum of unnormalised_weights : %f', sum(exp(hypothesisWeights)));

                    % normalized weights
                    [hypothesis_logWeights, ~] = normalizeLogWeights(hypothesis_logWeights);
                    % fprintf('\n Sum of normalised_weights : %f', sum(exp(hypothesis_logWeights)));

                    % for each measurement inside gate perform kalman filter update 
                    for index = 1: mk-1
                        hypothesisArray(index,1) = obj.density.update(state, z_inGate(:, index), measmodel);
                    end

                    % pruning and renormalising weights
                    [hypothesis_logWeights, hypothesisArray] = hypothesisReduction.prune(hypothesis_logWeights, ...
                    								   hypothesisArray, obj.reduction.w_min);
                    [hypothesis_logWeights, ~] = normalizeLogWeights(hypothesis_logWeights);
                    
                    % moment matching
                    [~ ,hypothesisArray] = hypothesisReduction.merge(hypothesis_logWeights ,hypothesisArray, ...
													10000000.0, obj.density);

                    % predict the next state
                    posteriorState = hypothesisArray(1,1);
				end
						
                % updated state variables 
                estimates{k}   = posteriorState.x;		
                estimates_x{k} = posteriorState.x;
                estimates_P{k} = posteriorState.P;
        		
                % predict the next state
                state = obj.density.predict(posteriorState, motionmodel);
            end 			
        end
        
        function [estimates_x, estimates_P] = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
            totalTrackTime = size(Z,1);
                        
            % placeholders for outputs
            estimates = cell(totalTrackTime, 1);
            estimates_x = cell(totalTrackTime, 1);
            estimates_P = cell(totalTrackTime, 1);
            
            % useful parameters
            log_wk_theta_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            log_wk_zero_factor  = log(1 - sensormodel.P_D);
            
            % initial state is considered posterior for first timestamp
            hypo_old = repmat(state,1,1);
            % since there's only 1 possible state it has log probability = log(1) = 0.0
            logweight_old = repmat([0.0],1,1); 

            % iterate through timestamps
            for k = 1 : totalTrackTime
    
                % get current timestep measurements
                zk = Z{k};
        
                % setting new variables as empty at every timestep
                hypo_new = [];
                logweight_new = [];

                % for every state in old hypothesis
                for hk = 1 : size(hypo_old, 1)

                    state_hk = hypo_old(hk, 1);
                    weight_hk = logweight_old(hk,1);
            
                    % 1. misdetection added as one state
                    hypo_new = [hypo_new; state_hk];
                    logweight_new = [logweight_new; weight_hk + log_wk_zero_factor];
        
                    % perform gating and find number of measurements inside limits
                    [z_inGate_hk, ~] = obj.density.ellipsoidalGating(state_hk, zk, measmodel, obj.gating.size);
                    mk_hk = size(z_inGate_hk, 2);

                    % 2. if there are measurements within gate of this hypothesis
                    if mk_hk > 0
        
                        % kalman filter update using valid measurements for each hypothesis
                        for index = 1 : mk_hk
                            posteriorState_index_hk = obj.density.update(state_hk, z_inGate_hk(:, index), measmodel);
                            hypo_new = [hypo_new; posteriorState_index_hk];
                        end
                        
                        % update weights for detected objects
                        likelihoodDensity = obj.density.predictedLikelihood(state_hk, z_inGate_hk, measmodel); 
                        logweight_new = [logweight_new; weight_hk + log_wk_theta_factor + likelihoodDensity];
                    end
                end
                            
                % 3. normalise hypothesis weights
                [logweight_new, ~] = normalizeLogWeights(logweight_new);
            
                % 4. prune small weights 
                [logweight_new, hypo_new] = hypothesisReduction.prune(logweight_new, hypo_new, obj.reduction.w_min);
                % renormalise weights
                [logweight_new, ~] = normalizeLogWeights(logweight_new);
                
                % 5. hyothesis merge 
                [logweight_new, hypo_new] = hypothesisReduction.merge(logweight_new, hypo_new, obj.reduction.merging_threshold, obj.density);
                [logweight_new, ~] = normalizeLogWeights(logweight_new);
                
                % 6. cap the number of hypothesis and renormalise weights
                [logweight_new, hypo_new] = hypothesisReduction.cap(logweight_new, hypo_new, obj.reduction.M);
                [logweight_new, ~] = normalizeLogWeights(logweight_new);
                
                % 7. extract object estimate  using most probable hypo
                [val, indices] = sort(logweight_new,'descend');
                estimates{k}   = hypo_new(indices(1)).x;
                estimates_x{k} = hypo_new(indices(1)).x;
                estimates_P{k} = hypo_new(indices(1)).P;                
                
                % 8. for each hypo, perform prediction		
                for idx = 1 : size(hypo_new,1)
                    hypo_new(idx,1) = obj.density.predict(hypo_new(idx,1), motionmodel);
                end
        
                hypo_old = hypo_new;
                logweight_old = logweight_new; 
            end			
        end 
    end
end