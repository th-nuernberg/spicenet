% function to parametrize the learning rate of the
% self-organizing-population-code network
function y = parametrize_learning_law(v0, vf, t0, tf, type)
y = zeros((tf-t0), 1);
t = 1:tf;
switch(type)
    case 'sigmoid'
        s = -floor(log10(tf))*10^(-(floor(log10(tf)))); % s<0 for decay
        p = abs(s*10^(floor(log10(tf)) + floor(log10(tf))/2));
        y = v0 - (v0)./(1+exp(s*(t-(tf/p)))) + vf;
        % plot(t, y, '.r'); box off; set(gcf, 'color', 'w'); title('sigmoid adaptation law');
    case 'invtime'
        B = (vf*tf - v0*t0)/(v0 - vf);
        A = v0*t0 + B*v0;
        y = A./(t+B);
        %plot(t, y, '.b'); box off; set(gcf, 'color', 'w'); title('inverse time adaptation law');
    case 'exp'
        if v0<1
            p = -log(v0);
        else
            p = log(v0);
        end
        y = v0*exp(-t./(tf/p));
%         plot(t, y, '.b'); box off; set(gcf, 'color', 'w'); title('exponential adaptation law');
end
end