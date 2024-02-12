% function to visualize network data at a given iteration in runtime
function id_maxv = visualize_runtime(populations, t)
set(gcf, 'color', 'white');
% extract the max weight on each row (if multiple the first one)
id_maxv = zeros(populations(1).lsize, 1);
for idx = 1:populations(1).lsize
    [~, id_maxv(idx)] = max(populations(1).Wcross(idx, :));
end
HL = (populations(1).Wcross)';
hndl1 = imagesc(HL, [0, max(HL(:))]); box off; colorbar;
xlabel('neuron index'); ylabel('neuron index'); title(sprintf('Hebbian connection matrix @ epoch %d', t));
% refresh graphics
set(hndl1, 'CData', HL);
drawnow;
end