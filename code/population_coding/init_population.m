% initialize a population
% encode a value in the population and display the preferred value and
% tuning curves of the population

% pop           - population struct
% enc_val       - encoded value in the population
% info          - tuning curves / encoded value selector flag

function y = init_population(pop, enc_val, info)
% init index
idx = 1;
switch info
    case 'tuning_curves'
        for idx=1:pop(idx).size
            plot(pop(idx).fi.p, pop(idx).fi.v);
            hold all;
        end;
        grid off;
        set(gca, 'Box', 'off');
        title('Tuning curves of the neural population');
        ylabel('Activity (spk/s)');
    case 'encoded_value'
        for idx=1:pop(idx).size
            % scale the firing rate to proper values and compute fi
            pop(idx).ri = gauss_val(enc_val, ...
                pop(idx).vi, ...
                pop(idx).sigma, ...
                pop(idx).max_rate) + ...
                pop(idx).eta;
            % rate should be positive althought noise can make small values
            % negative
            pop(idx).ri = abs(pop(idx).ri);
        end;
        % equally spaced receptive fields
        spacing = pop(idx).range / ((pop(idx).size-1)/2);
        % plot the noisy hill of population activity encoding the given value
        % index for neurons
        jdx = 1;
        for idx=-pop(idx).range:pop(idx).range
            % display on even spacing of the entire input domain
            if(rem(idx, spacing)==0)
                plot(idx, pop(jdx).ri, 'o');
                hold all;
                jdx = jdx+1;
            end;
        end;
        grid off;
        set(gca, 'Box', 'off');
        title(sprintf('Noisy activity of the population encoding the value %d', enc_val));
        ylabel('Activity (spk/s)');
        xlabel('Preferred value');
    case 'inferred_output'
        % equally spaced receptive fields
        spacing = pop(idx).range / ((pop(idx).size-1)/2);
        % plot the noisy hill of population activity
        jdx = 1;
        for idx=-pop(jdx).range:pop(jdx).range
            % display on even spacing of the entire input domain
            if(rem(idx, spacing)==0)
                plot(idx, pop(jdx).ri, 'o');
                hold all;
                jdx = jdx+1;
            end;
        end;
        % the encoded value in the output population is given by the embedded
        % function phi
        grid off;
        set(gca, 'Box', 'off');
        title(sprintf('Noisy activity of the population encoding the value of phi(x,y) after relaxation (t --> Inf)'));
        ylabel('Activity (spk/s)');
        xlabel('Preferred value');
end
    y = pop;
end