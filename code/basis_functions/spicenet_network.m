% The Multiple input Network with Intermefiate 
% Basis Function Neurons, IBFN

clc
close all
clear all

%% Parameter settings
N = 40;
Kw = 1;
sig_w = 0.37;    % In Rad
mu = 0.002;
S = 0.1;        % In Hzde
sig_c = 0.4;      % In radians
v = 1;
C = 1;
epoch = 10; count = 1;

Cr = 1;
Ce = 1;
Ca = 1;
Cnoise = 0.0;

xr = 1.2*pi;
xe = 0.3*pi;
xa = 1.8*pi;
noise1 = 0.5*pi;
noise2 = 1.5*pi;
noise3 = 0.8*pi;


% nr = (xr/(2*pi))*40
% ne = (xe/(2*pi))*40
% na = (xa/(2*pi))*40

j = 1:1:N;
l = 1:1:N;
m = 1:1:N;

[M L] = meshgrid(l,m);

% global A;       % Pattern of Actvity fo Intermediate Layer
% global Rr;      % Activity pattern vector for Xr population,
% global Re;      % Activity pattern vector for Xe population,
% global Ra;      % Activity pattern vector for Xa population,

Rr = zeros(1,N);   
Re = zeros(1,N);   
Ra = zeros(1,N);   
A = zeros(N,N);

% A = 3*rand(N,N);
%   as a formal rule: W is arranged as follows:
%       consisits of N-(N/2*N/2) Matrix, each one corresponds to.. 
%       the connection weights between jth Input neuron and intermediate
%       ;ayer
Wr_lmj = zeros(N, N, N);
We_lmj = zeros(N, N, N);
Wa_lmj = zeros(N, N, N);

%% Weight Setting 
sig = 2;
for i = 1:1:N
    J = ones(N,N) * i;

    C1 = ( cos( (2*pi/N)*(J-L) ) - 1 ) /  (sig_w^2);
    Wr_lmj(:,:,i) = 1*Kw*exp(1*C1);

    C2 = ( cos( (2*pi/N)*(J-M) ) - 1 ) /  ((sig_w)^2);
    We_lmj(:,:,i) = Kw*exp(1*C2);
 
    C3 = ( cos( (2*pi/N)*(J-(M + L)) ) - 1 ) / (sig_w^2);
    Wa_lmj(:,:,i) = Kw*exp(C3);
    
%     D1 = min( abs(J-L) , N - ( abs(J-L) ) );
%     Wr_lmj(:,:,i) = Kw*exp( (-D1/2).^2/(2*sig^2) );
% 
%     D2 = min( abs(J-M) , N - ( abs(J-M) ) );
%     We_lmj(:,:,i) = Kw*exp( (-D2/2).^2/(2*sig^2) );
%     
%     D3 = min( abs(J-L-M) , N - ( abs(J-L-M) ) );
%     Wa_lmj(:,:,i) = Kw*( exp((-D3/2)/1.0*sig_w).^2 );

%     Wa_lmj(:,:,i) = zeros(N,N);

%     load Wa_lmj;

end

%% Weight patterns between input populations and intermediate population 
figure;
% pause(15);
for k = 1:1:N
    h = surf(L, M, We_lmj(:,:,k) );
    refreshdata(h,'caller') % Evaluate y in the function workspace
    drawnow; 
    xlabel('Rr: L array Neurons');
    ylabel('Re: M array Neurons');
    title(['Weights of ' num2str(k) ...
        'th of Re Neurons to intermediate Layer'], 'FontSize', 16);
    pause(.01)
end

for k = 1:1:N
    h = surf( L, M, Wr_lmj(:,:,k) );
    refreshdata(h,'caller') % Evaluate y in the function workspace
    drawnow;
    xlabel('Rr: L array Neurons');
    ylabel('Re: M array Neurons');
    title(['Weights of ' num2str(k) ...
        'th Neuron of Rr population to intermediate Layer'], 'FontSize', 16);
    pause(0.01);
end

for k = 1:1:N
    h = surf( Wa_lmj(:,:,k) );
    refreshdata(h,'caller') % Evaluate y in the function workspace
    drawnow; pause(.1)
    xlabel('Rr: L array Neurons');
    ylabel('Re: M array Neurons');
    title(['Weights of ' num2str(k) ...
        'th Ra Neurons to intermediate Layer'], 'FontSize', 16);
     pause(.01)
end

%%  Activity Patern Initialization

Rr_0 = tuning_curve_noiseless(xr , sig_c, v, N, Cr);
Re_0 = tuning_curve_noiseless(xe , sig_c, v, N, Ce);
Ra_0 = tuning_curve_noiseless(xa , sig_c, v, N, Ca);
R_noise1 = tuning_curve_noiseless(noise1 , sig_c, v, N, Cnoise);
R_noise2 = tuning_curve_noiseless(noise2 , sig, v, N, Cnoise);
R_noise3 = tuning_curve_noiseless(noise3 , sig, v, N, Cnoise);

Rr_0 = Rr_0 + R_noise1;

% Rr_0 = Rr_0 + R_noise3 + R_noise1 + R_noise2;
% Re_0 = Re_0 + R_noise;

in = linspace(1/20,2,N);
figure; 
plot(in,Rr_0,'--rs','MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5);
hold on;
plot(in,Re_0,'--o','MarkerEdgeColor','k',...
                'MarkerFaceColor','y',...
                'MarkerSize',5);
hold on;
plot(in,Ra_0,'--.','MarkerEdgeColor','k',...
                'MarkerFaceColor','r',...
                'MarkerSize',15);
grid on;
%%  Network Evolotuon; Neurons Activity Exchange
Rr = Rr_0;
Re = Re_0;
Ra = Ra_0;

Lr_lm = zeros(N,N);
Le_lm = zeros(N,N);
La_lm = zeros(N,N);

% Matrixes containing the evolution of the activity patterns through
% epoches
Rr_M = zeros(N,epoch+1); Rr_M(:,1) = Rr';
Re_M = zeros(N,epoch+1); Re_M(:,1) = Re';
Ra_M = zeros(N,epoch+1); Ra_M(:,1) = Ra';
A_M = zeros(N,N,epoch+1);
A_M(:,:,1) = A;
Delta_A = zeros(epoch,1);

num_M = zeros(N,N,epoch);

c_d = 1;
while(epoch ~= 0)
    
    % Intermediate Activity Evolution -> e.g. A Matrix
    for k_l = 1:1:N
        for k_m = 1:1:N
            temp = zeros(N,1);
            temp(1:N,1) = Wr_lmj(k_l,k_m,:);
            Lr_lm(k_l,k_m) = Rr * temp;
            temp(1:N,1) = We_lmj(k_l,k_m,:);
            Le_lm(k_l,k_m) = Re * temp;
            temp(1:N,1) = Wa_lmj(k_l,k_m,:);
            La_lm(k_l,k_m) = Ra * temp;
        end
    end
    num = (Lr_lm + Le_lm + La_lm).^2;
    num_M(:,:,count) = sqrt(num);
    den = S + mu * ( sum( sum(num) ) ); 
    A = num/den;

    temp_Rr = zeros(1,N); 
    temp_Re = zeros(1,N); 
    temp_Ra = zeros(1,N);
    for j = 1:1:N
        temp_Rr(j) = ( sum(sum ( A .* Wr_lmj(:,:,j) )) )^2;
        temp_Re(j) = ( sum(sum ( A .* We_lmj(:,:,j) )) )^2;
        temp_Ra(j) = ( sum(sum ( A .* Wa_lmj(:,:,j) )) )^2;
    end
    Rr = temp_Rr / (S + mu*( sum(temp_Rr)) );
    Re = temp_Re / (S + mu*( sum(temp_Re)) );
    Ra = temp_Ra / (S + mu*( sum(temp_Ra)) );
    
    count = count + 1;
    epoch = epoch - 1;
    
    Rr_M(:,count) = Rr';
    Re_M(:,count) = Re';
    Ra_M(:,count) = Ra';
    A_M(:,:,count) = A;
    
    Delta_A(c_d) = sum(sum( A_M(:,:,count) ) ) - sum(sum( A_M(:,:,count-1) ) )
    c_d = c_d + 1;
end

%% Final Population tuning curves
figure; 
plot(in,Rr,'--rs','MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5);
hold on;
plot(in,Re,'--o','MarkerEdgeColor','k',...
                'MarkerFaceColor','y',...
                'MarkerSize',5);
hold on;
plot(in,Ra,'--.','MarkerEdgeColor','k',...
                'MarkerFaceColor','r',...
                'MarkerSize',15);
grid on;
%% Nework Activity Evolution Diagrams
figure;
% pause(1);
Xr = zeros(count,1);
Xe = zeros(count,1);
Xa = zeros(count,1);
for k = 1:1:count
    h1 = subplot(1,2,1); 
    plot(in,Rr_M(:,k),'--rs','MarkerEdgeColor','k',...
                    'MarkerFaceColor','g',...
                    'MarkerSize',3);
    hold on;
    plot(in,Re_M(:,k),'--o','MarkerEdgeColor','k',...
                    'MarkerFaceColor','y',...
                    'MarkerSize',3);
    hold on;
    plot(in,Ra_M(:,k),'--.','MarkerEdgeColor','k',...
                    'MarkerFaceColor','r',...
                    'MarkerSize',6);
    xlabel('The Angle as a factor of Pi in Radian','FontSize',18);
    ylabel('The Pattern of activity','FontSize',18);
    axis([0 2 0 70]); grid on;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear i
    Pr = sum( Rr_M(:,k)' .* exp(i*(2*pi/N)*(1:1:N)) );    
    Pe = sum( Re_M(:,k)' .* exp(i*(2*pi/N)*(1:1:N)) );    
    Pa = sum( Ra_M(:,k)' .* exp(i*(2*pi/N)*(1:1:N)) );    


    Xr(k) = angle(Pr)/pi;
    Xe(k) = angle(Pe)/pi;
    Xa(k) = angle(Pa)/pi;
    
    if(Xr(k) < 0 ) 
        Xr(k) = 2 + Xr(k);
    end
    if(Xe(k) < 0 ) 
        Xe(k) = 2 + Xe(k);
    end
    if(Xa(k) < 0 ) 
        Xa(k) = 2 + Xa(k);
    end

    title(['Xr = ' num2str(Xr(k)*pi) '  Xe = ' num2str(Xe(k)) '  Xa = ' ...
        num2str(Xa(k)*pi)],'FontSize',20);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    h2 = subplot(1,2,2); pcolor(L,M,A_M(:,:,k));
    title(['xr_0 = ' num2str(xr) '  xe_0 = ' num2str(xe) '  xa_0 = ' ...
        num2str(xa)],'FontSize',20);
    refreshdata(h1,'caller') % Evaluate y in the function workspace
    refreshdata(h2,'caller') % Evaluate y in the function workspace
    drawnow;  

    pause(0.5);
end


% clear i
% Pr = sum( Rr .* exp(i*(2*pi/N)*(1:1:N)) );    
% Pe = sum( Re .* exp(i*(2*pi/N)*(1:1:N)) );    
% Pa = sum( Ra .* exp(i*(2*pi/N)*(1:1:N)) );    
% 
% Xr = angle(Pr);
% Xe = angle(Pe);
% Xa = angle(Pa);
% 
% [xr Xr]*180/pi
% [xe Xe]*180/pi
% [xa Xa]*180/pi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Nework Activity Evolution Diagrams
in = linspace(1/20,2,N);

for k = 1:1:count
%     figure;
%     subplot(1,2,1); 
%     plot(in,Rr_M(:,k),'--rs','MarkerEdgeColor','k',...
%                     'MarkerFaceColor','g',...
%                     'MarkerSize',3);
%     hold on;
%     plot(in,Re_M(:,k),'--o','MarkerEdgeColor','k',...
%                     'MarkerFaceColor','y',...
%                     'MarkerSize',3);
%     hold on;
%     plot(in,Ra_M(:,k),'--.','MarkerEdgeColor','k',...
%                     'MarkerFaceColor','r',...
%                     'MarkerSize',6);
%     axis([0 2 0 70]); grid on;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear i
    Pr = sum( Rr_M(:,k)' .* exp(i*(2*pi/N)*(1:1:N)) );    
    Pe = sum( Re_M(:,k)' .* exp(i*(2*pi/N)*(1:1:N)) );    
    Pa = sum( Ra_M(:,k)' .* exp(i*(2*pi/N)*(1:1:N)) );    

    Xr1 = angle(Pr)/pi;
    Xe1 = angle(Pe)/pi;
    Xa1 = angle(Pa)/pi;
    
    if(Xr1 < 0) 
        Xr1 = 2 + Xr1;
    end
    if(Xe1 < 0) 
        Xe1 = 2 + Xe1;
    end
    if(Xa1 < 0) 
        Xa1 = 2 + Xa1;
    end

    title(['Xr = ' num2str(Xr1) '  Xe = ' num2str(Xe1) '  Xa = ' ...
        num2str(Xa1)],'FontSize',20);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    subplot(1,2,2); surf(L,M,A_M(:,:,k));
    title(['xr_0 = ' num2str(xr) '  xe_0 = ' num2str(xe) '  xa_0 = ' ...
        num2str(xa)],'FontSize',20);
end

log_A = log10(Delta_A);
figure; plot((1:count-1)',log_A,'--o');

figure; 
plot(0:1:count-1,Xr,'--o'); 
hold on; grid on;
plot(0:1:count-1,Xe,'--.'); 
hold on;
plot(0:1:count-1,Xa,'--*'); 
hold on;
plot(0:1:count-1,Xr+Xe,'--rs'); 

figure; plot3(Xr,Xe,Xa,'--o'); hold on; 
    plot3(Xr(1),Xe(1),Xa(1),'--o','MarkerEdgeColor','k',...
                    'MarkerFaceColor','r',...
                    'MarkerSize',6);
hold on;
[X Y] = meshgrid(linspace(0,2*pi,N));
mesh(X,Y,X+Y); grid on;

grid on;
xlabel('Xr'); ylabel('Xe'); zlabel('Xa');
% axis([0 360 0 360 0 360]);
