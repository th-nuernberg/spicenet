% function to visualize network data at a given iteration in runtime
function id_maxv = visualize_results(sensory_data, populations, learning_params)
figure;
set(gcf, 'color', 'white');
% sensory data
subplot(4, 1, [1 2]);
plot(sensory_data.x, sensory_data.y, '.g'); xlabel('X'); ylabel('Y'); box off;
title('Encoded relation');
% learned realtionship encoded in the Hebbian links
subplot(4, 1, [3 4]);
% extract the max weight on each row (if multiple the first one)
id_maxv = zeros(populations(1).lsize, 1);
for idx = 1:populations(1).lsize
    [~, id_maxv(idx)] = max(populations(1).Wcross(idx, :));
end
imagesc((populations(1).Wcross)', [0, 1]); box off; colorbar;
% for 3rd order or higher order add some overlay
% imagesc(rot90(populations(1).Wcross), [0, 1]); box off; colorbar;
% hold on; plot(1:populations(1).lsize, rot90(id_maxv') ,'r', 'LineWidth', 2);
% hold on; plot(1:populations(1).lsize, rot90(id_maxv') ,'ok', 'MarkerFaceColor','k');
xlabel('neuron index'); ylabel('neuron index');
% learning parameters in different figures
figure; set(gcf, 'color', 'w');
plot(learning_params.alphat, 'k', 'LineWidth', 3); box off; ylabel('SOM Learning rate'); 
xlabel('SOM training epochs'); 
figure; set(gcf, 'color', 'w');
plot(parametrize_learning_law(populations(1).lsize/2, 1, learning_params.t0, learning_params.tf_learn_in, 'invtime'), 'k', 'LineWidth', 3); 
box off; ylabel('SOM neighborhood size'); xlabel('SOM training epochs'); 
% hebbian learning 
figure; set(gcf, 'color', 'w');
etat = parametrize_learning_law(0.1, 0.001, learning_params.t0, learning_params.tf_learn_cross, 'invtime');
plot(etat, 'm', 'LineWidth', 3); box off; ylabel('Hebbian Learning rate'); xlabel('Hebbian learning epochs'); 
% % show the topology learning (self organization)
figure; set(gcf, 'color', 'w');
subplot(2,1,1);
plot(populations(1).Winput, '.g'); xlabel('neuron index in pop 1'); ylabel('preferred value'); box off;
subplot(2,1,2);
plot(populations(2).Winput, '.b'); xlabel('neuron index in pop 2'); ylabel('preferred value'); box off;
end