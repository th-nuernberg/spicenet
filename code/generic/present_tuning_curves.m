% function to plot the learned tuning curves and the probability density
% function of the input data
function present_tuning_curves(pop, sdata)
figure; set(gcf, 'color', 'w');
subplot(4,1,1);
% plot the probability distribution of the input data to motivate the
% density of the learned tuning curves, this is the sensory prior p(s)
switch pop.idx
    case 1
        hist(sdata.x, 50); box off;
    case 2
        hist(sdata.y, 50); box off;
end
xlabel(sprintf('range pop %d ', pop.idx)); ylabel('input values distribution');
hndl = subplot(4, 1, 2);
% compute the tuning curve of the current neuron in the population
% the equally spaced mean values
x = linspace(-sdata.range, sdata.range, pop.lsize);

% % DISPLAY FOR ALL NEURONS
% for each neuron in the current population compute the receptive field
for idx = 1:pop.lsize
    % extract the preferred values (wight vector) of each neuron
    v_pref = pop.Winput(idx);
    fx = exp(-(x - v_pref).^2/(2*pop.s(idx)^2));
    plot(1:pop.lsize, fx, 'LineWidth', 3); hold all;
end
pop.Winput = sort(pop.Winput); box off;
ax1_pos = get(hndl, 'Position'); set(hndl, 'XTick', []); set(hndl, 'XColor','w');
ax2 = axes('Position',ax1_pos,'XAxisLocation','bottom','Color','none','LineWidth', 3);
set(hndl, 'YTick', []); set(hndl, 'YColor','w');
set(ax2, 'XTick', pop.Winput); set(ax2, 'XTickLabel', []);
set(ax2, 'XLim', [ min(x), max(x)]);
set(ax2, 'XTickLabelRotation', 90);
xlabel('neuron preferred values'); ylabel('learned tuning curves shapes');

% the density of the tuning curves (density function) - should increase
% with the increase of the distribution of sensory data (directly proportional with p(s))
% stimuli associated with the peaks of the tuning curves
subplot(4,1,3);
hist(pop.Winput, 50); box off;
xlabel(sprintf('range pop %d ', pop.idx)); ylabel('# of allocated neurons');
% the shape of the tuning curves (shape functions) - should increase with
% values distribution decrease (inverse proportionally with sensory prior p(s))
% measured as the full width at half maximum of the tuning curves
subplot(4,1,4);
plot(pop.s, '.r'); box off;
xlabel('neuron index'); ylabel('width of tuning curves');


% DISPLAY ONLY SOME NEURONS
figure;
hndl = subplot(1,1,1);
v_pref = sort(pop.Winput);
% for each neuron in the current population compute the receptive field
% select some tuning curves to plot
pref = [1, 6, 13, 40, 45, 85, 90, 99];
for idx = 1:length(pref)
    idx_pref = pref(idx);
    % extract the preferred values (weight vector) of each neuron
    fx = exp(-(x - v_pref(idx_pref)).^2/(2*pop.s(idx_pref)^2));
    plot(1:pop.lsize, fx,'LineWidth', 3); hold all;
end
ax1_pos = get(hndl, 'Position'); set(hndl, 'XTick', []); set(hndl, 'XColor','w');
ax2 = axes('Position',ax1_pos,'XAxisLocation','bottom','Color','none','LineWidth', 3);
set(hndl, 'YTick', []); set(hndl, 'YColor','w');
v_pref_idx = zeros(1, length(pref));
for idx = 1:length(pref)
    v_pref_idx(idx) = v_pref(pref(idx)+1);
end
set(ax2, 'XTick', v_pref_idx); set(ax2, 'XTickLabel', v_pref_idx);
set(ax2, 'XLim', [min(x), max(x)]);
set(ax2, 'XTickLabelRotation', 90);
xlabel(ax2, 'preferred value');
ylabel('learned tuning curves shapes');
ax3 = axes('Position',ax2.Position,...
    'XAxisLocation','top',...
    'Color','none','LineWidth', 0.01);
set(ax3, 'XTick', v_pref_idx); 
set(ax3, 'Ticklength', [0 0]); 
set(ax3, 'XTickLabel', pref);
set(ax3, 'XLim', [min(x), max(x)]);
xlabel(ax3, 'neuron index');

end