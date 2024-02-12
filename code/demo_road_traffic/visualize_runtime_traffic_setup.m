% function to visualize network data at a given iteration in runtime
function visualize_runtime_traffic_setup(uon_sensory_data,sensory_data, populations, d, sensor1, sensor2, img)
    set(gcf, 'color', 'w');   
    % overlay scenario
    stp = subplot(8, 3, [1 9]);
   
    image( imread(img) ); box off; axis off;
    switch sensor1
        case 'NO2'
            color1 = '.g';
        case 'VehicleCount'
            color1 = '.r';
        case 'Humidity'
            color1 = '.b';
    end
    set(stp,'LooseInset',get(stp,'TightInset'));
    
    % sensory data 1
    d1 = subplot(8, 3, 10);
    acth3 = plot(uon_sensory_data.x(d), color1, 'MarkerSize', 10); box off; hold on;
    set(d1,'LooseInset',get(d1,'TightInset'));
    set(d1, 'YtickLabel',[]);
    set(d1, 'XtickLabel',[]);
    ylabel(sensor1); 
    title(sprintf('Sensory data sample %d', d));
    
    % sample pair
    d2 = subplot(8, 3, [11 15]);
    set(d2,'LooseInset',get(d2,'TightInset'));
    acth5 = plot(sensory_data.x(d), sensory_data.y(d), 'ok', 'MarkerEdgeColor', 'k', 'MarkerSize', 10); 
    hold on; plot(sensory_data.x, sensory_data.y, '.g'); box off;
    xlabel(sensor1); ylabel(sensor2);
    set(d2, 'YtickLabel',[]);
    set(d2, 'XtickLabel',[]);
    title('Input data');
    if (strcmp(sensor1,'NO2') == 1 && strcmp(sensor2,'VehicleCount') == 1)
        legend('Current sample',sprintf('Underlying relation - \n UNKNOWN'), 'Location','SouthEast'); legend boxoff;
    else
        legend('Current sample',sprintf('Underlying relation - \n UNKNOWN'), 'Location','SouthWest'); legend boxoff;
    end
    
    % sensory data 2
    switch sensor2
        case 'NO2'
            color2 = '.g';
        case 'VehicleCount'
            color2 = '.r';
        case 'Humidity'
            color2 = '.b';
    end
    d3 = subplot(8, 3, 13);
    set(d3,'LooseInset',get(d3,'TightInset'));
    acth4 = plot(uon_sensory_data.y(d), color2, 'MarkerSize', 10); box off; hold on;
    set(d3, 'YtickLabel',[]);
    set(d3, 'XtickLabel',[]);
    ylabel(sensor2);
            
    % hebbian links between populations containing the learnt relation
    hpc1 = subplot(8, 3, [16 24]);
    set(hpc1,'LooseInset',get(hpc1,'TightInset'));
    ax1=get(hpc1,'position'); % Save the position as ax
    set(hpc1,'position',ax1); % Manually setting this holds the position with colorbar
    acth9 = imagesc((flipud(populations(1).Wext'))); caxis([0, max(populations(1).Wext(:))]); 
    %colorbar;
    box off; grid off;set(gca,'XAxisLocation','top');
    set(hpc1, 'YtickLabel',[]);
    set(hpc1, 'XtickLabel',[]);
    title(sprintf('Learnt relation: %s - %s',sensor1, sensor2)); 
    
    % refresh visualization
    set(acth3, 'XData', d);
    set(acth3, 'YData', uon_sensory_data.x(d));
    set(acth4, 'XData', d);
    set(acth4, 'YData', uon_sensory_data.y(d));
    set(acth5, 'XData', sensory_data.x(d));
    set(acth5, 'YData', sensory_data.y(d));
    set(acth9, 'CData', flipud(populations(1).Wext'));
    
    drawnow;
end