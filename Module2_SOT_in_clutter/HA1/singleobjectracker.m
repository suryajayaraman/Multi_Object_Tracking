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
        
        function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
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
            
            total_track_time = length(Z);  
            %N = numel(Z);
            N = total_track_time;
            %disp(class(Z));
            % https://in.mathworks.com/matlabcentral/answers/106341-how-to-get-type-of-a-variable-in-matlab#:~:text=To%20get%20the%20data%20type,use%20the%20%E2%80%9Cclass%E2%80%9D%20function.&text=To%20determine%20if%20a%20variable,use%20the%20%E2%80%9Cisa%E2%80%9D%20function.&text=For%20a%20list%20of%20functions,%2C%20see%20%E2%80%9Cis*%E2%80%9D.
            
            %disp(class(Z{1})); returns 'double' type, a matrix
            % disp(size(Z(1))); returns 'cell',  type, a cell array
            %disp(size(Z{1})); % returns 2 x N size matrix
            %disp(size(Z(1))); % returns 1 x 1 size cell array
            
            % initialise output
            estimates = cell(N,1);
            
            % first predicted state is the prior distribution
            % state_pred = state;
            
            % weights factor
            w_theta_k_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            w_theta_0_factor = log(1 - sensormodel.P_D);
            
            for i=1:N
                z = Z{i};
                
                % Apply gating
                [z_ingate, ~] = obj.density.ellipsoidalGating(state, z, measmodel, obj.gating.size);
                
                % number of hypothesis
                mk = size(z_ingate,2) + 1;
                
                % no valid measurements within gate
                if(mk==1)
                    state_upd = state;
                    
                else
                    % calculate predicted likelihood = N(z; zbar, S) => scalar value
                    log_likelihood = obj.density.predictedLikelihood(state,z_ingate,measmodel);

                    % EVALUATE HYPOTHESIS
                    % detection and missdetection
                    w_theta_k = exp(w_theta_k_factor + log_likelihood);
                    w_theta_0 = exp(w_theta_0_factor);
                    
                    % measurement update
                    [max_w_theta, max_theta] = max(w_theta_k);
                    if(w_theta_0 > max_w_theta)
                        state_upd = state;
                    else
                        z_NN = z_ingate(:,max_theta);    % nearest neighbour measurement
                        state_upd = obj.density.update(state, z_NN, measmodel);
                    end
                end
                
                % updated stat 
                estimates{i} = state_upd.x;
                
                % predict the next state
                state = obj.density.predict(state_upd, motionmodel);
            end
        end
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
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
            N = length(Z);
            
            % initialise output
            estimates = cell(N,1);
            
            % weights factor
            w_theta_k_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            w_theta_0_factor = log(1 - sensormodel.P_D);
            
            			
            for i=1:N
            	z = Z{i};
            	% fprintf('\n Timestamp : %d', i);
            	% fprintf('\n Before gating there are %d measurements', size(z,2));
            	
            	% Apply gating
            	[z_ingate, ~] = obj.density.ellipsoidalGating(state, z, measmodel, obj.gating.size);
            	% fprintf('\n After gating there are %d measurements', size(z_ingate,2));
            
            	% number of hypothesis
            	mk = size(z_ingate,2) + 1;
            	
            	% object is undetected
            	if mk ==1
            		%fprintf('\n object is undetected');
            		state_upd = state;
            	
            	% process measurements
            	else
            		
            		% hypothesis cell array intialise
            		multiHypotheses = repmat(state, mk,1);
            		
            		% missed_detection_hypo is the last row
            		multiHypotheses(mk,1) = state;
            		
            		% calculate predicted likelihood = N(z; zbar, S) => scalar value
            		log_likelihood = obj.density.predictedLikelihood(state,z_ingate,measmodel);
            		
            		% detection and missdetection weights
            		log_w_theta_k = w_theta_k_factor + log_likelihood;
            		log_w_theta_0 = w_theta_0_factor;
            		unnormalised_log_weights = [log_w_theta_k; log_w_theta_0];
            		% fprintf('\n Sum of unnormalised_weights : %f', sum(exp(unnormalised_log_weights)));
            		
            		[normalised_log_weights, ~] = normalizeLogWeights(unnormalised_log_weights);
            		% fprintf('\n Sum of normalised_weights : %f', sum(exp(normalised_log_weights)));
            				
            		% detection hypothesis
            		for index = 1: mk-1
            			hypo_thetak = obj.density.update(state, z_ingate(:, index), measmodel);
            			multiHypotheses(index,1) = hypo_thetak;		
            		end
            		% fprintf('\n Size of normalised_log_weights %s', mat2str(size(normalised_log_weights)));
            		% fprintf('\n Size of multiHypotheses %s', mat2str(size(multiHypotheses)));
            %                     fprintf('\n Size of multiHypotheses(2) %s', mat2str(size(multiHypotheses(2))));
            %                     fprintf('\n Size of state vector is %s', mat2str(size(multiHypotheses(1).x)));
            		
            		% pruning and renormalising weights
            		[hypothesesWeight,multiHypotheses] = hypothesisReduction.prune(normalised_log_weights, multiHypotheses, obj.reduction.w_min);
            		[normalised_log_weights, ~] = normalizeLogWeights(hypothesesWeight);
            													
            		% moment matching
            		[~ ,multiHypotheses] = hypothesisReduction.merge(normalised_log_weights ,multiHypotheses, ...
            													10000000.0, obj.density);
            		state_upd = multiHypotheses(1,1);
            		% state_upd = obj.density.momentMatching(normalised_log_weights, multiHypotheses);
            	end
            					
            	% updated estimates array
            	estimates{i} = state_upd.x;
            	
            	% predict the next state
            	state = obj.density.predict(state_upd, motionmodel);
            end 
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
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
            % ### Constants for the algorithm
            N = length(Z);
            
            % initialise output
            estimates = cell(N,1);
            
            % weights factor
            w_theta_k_factor = log(sensormodel.P_D / sensormodel.intensity_c);
            w_theta_0_factor = log(1 - sensormodel.P_D);
            
            hypo_old = repmat(state,1,1);
            weight_old = repmat([0.0],1,1); %log(1) = 0.0
            
            % iterating throught timesteps
            for i = 1:N
            	z = Z{i};
            	%fprintf('\n Timestamp : %d', i);
            	%fprintf('\n Before gating there are %d measurements', size(z,2));
            
            	% setting new variables as empty at every timestep
            	hypo_new = [];
            	weight_new = [];
            
            	% number of hypothesis from previous timestep
            	Hk_old = size(hypo_old,1);
            
            	% iterating through each hypothesis
            	for hk = 1: Hk_old
            		state_hk = hypo_old(hk,1);
            		weight_hk = weight_old(hk,1);
            		
            		% 1. for each hypothesis create missed detection hypo
            		hypo_new = [hypo_new; state_hk];
            		weight_new = [weight_new; weight_hk + w_theta_0_factor];
            		
            		% Apply gating
            		[z_ingate, ~] = obj.density.ellipsoidalGating(state_hk, z, measmodel, obj.gating.size);
            		%fprintf('\n After gating there are %d measurements', size(z_ingate,2));
            
            		% number of hypothesis
            		mk_hk = size(z_ingate,2);
            		
            		% 2. for each hypo ellipsoid gating and only create detection hypo for that valid hypo only
            		if mk_hk > 0
            			% adding measurements to current hypothesis
            			for index = 1: mk_hk
            				hypo_thetak = obj.density.update(state_hk, z_ingate(:, index), measmodel);
            				hypo_new = [hypo_new; hypo_thetak];
            			end
            			
            			% calculating weights for the measurements
            			% calculate predicted likelihood = N(z; zbar, S) => scalar value
            			log_likelihood = obj.density.predictedLikelihood(state_hk,z_ingate,measmodel);
            			log_w_theta_k = w_theta_k_factor + log_likelihood;
            			weight_new = [weight_new; log_w_theta_k + weight_hk];
            		end
            	end
            	
            	% 3. normalise hypothesis weights
            	[weight_new, ~] = normalizeLogWeights(weight_new);
            
            	% 4. prune small weights 
            	[weight_new, hypo_new] = hypothesisReduction.prune(weight_new, hypo_new, obj.reduction.w_min);
            	% renormalise weights
            	[weight_new, ~] = normalizeLogWeights(weight_new);
            	
            	% 5. hyothesis merge 
            	[weight_new, hypo_new] = hypothesisReduction.merge(weight_new, hypo_new, obj.reduction.merging_threshold, obj.density);
                [weight_new, ~] = normalizeLogWeights(weight_new);
                
            	% 6. cap the number of hypothesis and renormalise wieghts
            	[weight_new, hypo_new] = hypothesisReduction.cap(weight_new, hypo_new, obj.reduction.M);
            	[weight_new, ~] = normalizeLogWeights(weight_new);
            	
            	% 7. extract object estimate  using most probable hypo
            	[~, indices] = sort(weight_new,'descend');
            	estimates{i} = hypo_new(indices(1)).x;
            			
            	% 8. for each hypo, perform prediction		
            	for idx = 1 : size(hypo_new,1)
            		hypo_new(idx,1) = obj.density.predict(hypo_new(idx,1), motionmodel);
            	end
            	hypo_old = hypo_new;
            	weight_old = weight_new; 
            end
        end
        
    end
end