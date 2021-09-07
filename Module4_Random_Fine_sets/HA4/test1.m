classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
           
        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
            
        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            
            estimates = [];

        end
        
    end
    
end

