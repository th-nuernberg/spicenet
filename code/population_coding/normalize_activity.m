% normalize an activity matrix within the min and max firing rates
function y = normalize_activity(matrix, minimum, maximum)
mat_size = size(matrix);
mat_min = min(min(matrix));
mat_max = max(max(matrix));
y = zeros(mat_size(1), mat_size(2));
        for r = 1:mat_size(1)
            for c = 1:mat_size(2)
                matrix(r,c) = minimum + ...
                           (matrix(r, c) - mat_min)*...
                           ((maximum - minimum)/(mat_max - mat_min));
                y(r, c) = matrix(r, c);
            end
        end
end
