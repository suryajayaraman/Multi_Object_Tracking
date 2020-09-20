%Script to animated results for SOT 
% estimator = "NN";
estimator = "PDA";
% estimator = "GSF";

%% save options
write_video = 0;
write_image = 0;

%%
switch estimator
    case 'NN'
        est_x = NN_estimated_state;
        est_P = nn_est_P;
        plot_title = "Single object tracking using NN";
        
    case 'PDA'
        est_x = PDA_estimated_state;
        est_P = pda_est_P;
        plot_title = "Single object tracking using PDA";
        
    case 'GSF'
        est_x = GS_estimated_state;
        est_P = gsf_est_P;
        plot_title = "Single object tracking using GSF";
end

% constants for animation
x_limit = 100.0;
y_limit = 100.0;
wait_time = 0.2;
history = 20;        

% figure1=figure('Position', [1000, 1000, 1000, 1000]);
      
% % zoomed in plot
% P1 = subplot(1,2,1);    
% 
% % overall plot
% P2 = subplot(1,2,2);
% hold on 
% plot(true_state(1,:), true_state(2,:), 'g','Linewidth', 1);
% plot(est_x(1,:), est_x(2,:), 'b' , 'Linewidth', 1);


%% Save filter performance as video

if write_video == 1
    myVideo = VideoWriter( plot_title + '.avi'); %open video file
    myVideo.FrameRate = round(1/wait_time);  %can adjust this, 5 - 10 works well for me
    open(myVideo)
end


for k = 1:length(true_state)

    start_index = max(k-history, 1);
    sigmaEllipse = sigmaEllipse2D( est_x(1:2,k), est_P{k}(1:2, 1:2));
    
    %%%%%%%%%%% SUBPLOT 1 %%%%%%%%%%% 
    clf;
%     set(gcf, 'Position',  [100, 100, 800, 800]);
    % plot ground truth and estimated values        
    plot(true_state(1, start_index: k), true_state(2, start_index: k), 'Linewidth', 2);
    hold on
    plot(est_x(1, start_index: k), est_x(2,  start_index: k), 'Linewidth', 2);

    % plot variances
    h = fill(sigmaEllipse(1,:), sigmaEllipse(2,:), 'c'); 
    h.FaceAlpha = 0.25;  % for transparency
    
    legend('Ground truth','Estimated output using ' + estimator, estimator + " variance")
    % set axis to roi
    axis([true_state(1,k) - x_limit, true_state(1,k) + x_limit, ...
          true_state(2,k) - y_limit, true_state(2,k) + y_limit]);
    
    title(plot_title);
    xlabel('X(metres)');
    ylabel('Y(metres)');
    
    grid on;
    axis square;
    
    if write_video == 1
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
    end
    
%     %%%%%%%%%%% SUBPLOT 2 %%%%%%%%%%% 
%     % plot ground truth values
%     plot(P2, true_state(1,k), true_state(2,k), 'Color', 'g', 'Marker', 'x', 'Linewidth', 2);
%     % plot estimated results
%     plot(P2, est_x(1,k), est_x(2,k), 'Color', 'b', 'Marker', 'o', 'Linewidth', 2);
        
    % sleep for some time b/w plots
    pause(wait_time);
end

if write_video == 1
    close(myVideo)
end

%% Overall estimator plot
figure();
plot(true_state(1,:), true_state(2,:), 'g','Linewidth', 2)
hold on
plot(est_x(1,:), est_x(2,:), 'b' , 'Linewidth', 1)
title(plot_title);
xlabel('x (m)')
ylabel('y (m)')
grid on
axis square
legend('Ground truth','Estimated output using ' + estimator)

set(gca,'FontSize',12)

%% save final plot as figure
if write_image ==1
    saveas(gcf, estimator + '_output.png')
end