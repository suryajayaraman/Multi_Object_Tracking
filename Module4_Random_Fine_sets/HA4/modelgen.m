classdef modelgen
    %MODELGEN is a class used to create the tracking model
    
    methods (Static)
        function obj = sensormodel(P_D,lambda_c,range_c)
            %SENSORMODEL creates the sensor model
            %INPUT:  P_D: object detection probability --- scalar
            %        lambda_c: average number of clutter measurements per
            %        time scan, Poisson distributed --- scalar 
            %        range_c: range of surveillance area --- if 2D model: 2
            %        x 2 matrix of the form [xmin xmax;ymin ymax]; if 1D
            %        model: 1 x 2 vector of the form [xmin xmax]  
            %OUTPUT: obj.pdf_c: uniform clutter density --- scalar
            %        obj.P_D: same as P_D 
            %        obj.lambda_c: same as lambda_c
            %        obj.range_c: same as range_c
            %        obj.intensity_c: uniform clutter intensity --- scalar            
            if size(range_c,1) == 1
                volume = abs(range_c(2) - range_c(1));
            else
                volume = abs(range_c(1,2) - range_c(1,1)) * abs(range_c(2,2) - range_c(2,1));
            end
            
            if volume < 0.0001
                volume = 0.0001;
            end
            %disp(volume);
			obj.P_D = P_D;
            obj.lambda_c = lambda_c;
            obj.range_c = range_c;
            obj.pdf_c = 1.0 / volume; 
            obj.intensity_c = obj.lambda_c * obj.pdf_c;
        end
        
        function obj = groundtruth(nbirths,xstart,tbirth,tdeath,K)
            %GROUNDTRUTH specifies the parameters to generate groundtruth
            %INPUT:  nbirths: number of objects hypothesised to exist from
            %        time step 1 to time step K --- scalar 
            %        xstart: object initial state --- (object state
            %        dimension) x nbirths matrix 
            %        tbirth: object birth (appearing) time --- (total number
            %        of objects existed in the scene) x 1 vector  
            %        tdeath: the last time the object exists ---  (total number
            %        of objects existed in the scene) x 1 vector 
            %        K: total tracking time --- scalar
            
            obj.nbirths = nbirths;
            obj.xstart = xstart;
            obj.tbirth = tbirth;
            obj.tdeath = tdeath;
            obj.K = K;
        end       
    end
end
