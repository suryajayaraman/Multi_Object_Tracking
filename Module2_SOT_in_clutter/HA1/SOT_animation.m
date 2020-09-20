%% Script to animated results for SOT 
%% save options
write_video = 1;
% write_image = 0;

%% constants for animation
x_limit = 100.0;
y_limit = 100.0;
wait_time = 0.2;
history = 20;    
plot_title = "SOT algorithm comparison";
face_alpha = 0.25;

% figure to plot
% fig1 = figure('name', 'f1', 'Position', [100, 100, 1000, 1000]);
% ax1 = axes('parent', fig1);

%% Save filter performance as video
if write_video == 1
    myVideo = VideoWriter( "SOT_animation_results/" + plot_title + '.avi'); %open video file
    myVideo.FrameRate = round(1/wait_time);  %can adjust this, 5 - 10 works well for me
    open(myVideo)
end

%% iterate through time
for k = 1:length(true_state)

    start_index = max(k-history, 1);
    
    % clear previous time plots
    clf;
    
    % plot ground truth and estimated values        
    plot(true_state(1, start_index: k), true_state(2, start_index: k), 'Linewidth', 2);
    hold on
    
    %%%%%%%%%%% NN estimated state and variance %%%%%%%%%%% 
    sigmaEllipse_nn = sigmaEllipse2D( NN_estimated_state(1:2,k), nn_est_P{k}(1:2, 1:2));
    plot(NN_estimated_state(1, start_index: k), NN_estimated_state(2,  start_index: k), 'Linewidth', 2);
    fill(sigmaEllipse_nn(1,:), sigmaEllipse_nn(2,:), 'c', 'FaceAlpha', face_alpha); 
    
    %%%%%%%%%%% PDA estimated state and variance %%%%%%%%%%% 
    sigmaEllipse_pda = sigmaEllipse2D( PDA_estimated_state(1:2,k), pda_est_P{k}(1:2, 1:2));
    plot(PDA_estimated_state(1, start_index: k), PDA_estimated_state(2,  start_index: k), 'Linewidth', 2);
    fill(sigmaEllipse_pda(1,:), sigmaEllipse_pda(2,:), 'y', 'FaceAlpha', face_alpha); 

    %%%%%%%%%%% GSF estimated state and variance %%%%%%%%%%% 
    sigmaEllipse_gsf = sigmaEllipse2D( GS_estimated_state(1:2,k), gsf_est_P{k}(1:2, 1:2));
    plot(GS_estimated_state(1, start_index: k), GS_estimated_state(2,  start_index: k), 'Linewidth', 2);
    fill(sigmaEllipse_gsf(1,:), sigmaEllipse_gsf(2,:), 'm', 'FaceAlpha', face_alpha); 

    legend('Ground truth' , ...
           'NN estimator' , 'NN variance',  ...
           'PDA estimator', 'PDA variance', ...
           'GSF estimator', 'GSF variance', ...
           'Location'     ,'best')
    
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

    % sleep for some time b/w plots
    pause(wait_time);
end

if write_video == 1
    close(myVideo)
end

%% Overall estimator plot
% figure();
% plot(true_state(1,:), true_state(2,:), 'g','Linewidth', 2)
% hold on
% plot(NN_estimated_state(1,:), NN_estimated_state(2,:), 'b' , 'Linewidth', 1)
% title(plot_title);
% xlabel('x (m)')
% ylabel('y (m)')
% grid on
% axis square
% legend('Ground truth','NN Estimated output')
% set(gca,'FontSize',12)
% 
% figure();
% plot(true_state(1,:), true_state(2,:), 'g','Linewidth', 2)
% hold on
% plot(PDA_estimated_state(1,:), PDA_estimated_state(2,:), 'b' , 'Linewidth', 1)
% title(plot_title);
% xlabel('x (m)')
% ylabel('y (m)')
% grid on
% axis square
% legend('Ground truth','PDA Estimated output')
% set(gca,'FontSize',12)
% 
% figure();
% plot(true_state(1,:), true_state(2,:), 'g','Linewidth', 2)
% hold on
% plot(GS_estimated_state(1,:), GS_estimated_state(2,:), 'b' , 'Linewidth', 1)
% title(plot_title);
% xlabel('x (m)')
% ylabel('y (m)')
% grid on
% axis square
% legend('Ground truth','GSF Estimated output')
% set(gca,'FontSize',12)

% %% save final plot as figure
% if write_image ==1
%     saveas(gcf, estimator + '_output.png')
% end