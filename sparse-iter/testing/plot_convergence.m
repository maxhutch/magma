function plot_convergence( write )


% This script visualizes the convergence for data stored in convergence.m
% For write == 1, the plot is written to disc as pdf.

% plot defaults
LW = 'linewidth'; lw = 1;
FS = 'fontsize'; fs = 14;
MS = 'markersize'; ms = 100;
MC = 'markerfacecolor'; mc = 'auto';

eval('convergence');

h = figure(1);

semilogy(...
     data(:,1),data(:,2), 'r-', ...
     LW, lw, MC, mc);
hold on;
ylabel('Residual norm',FS,fs);
xlabel('Number of iterations',FS,fs)
set(gca,FS,fs),grid on
%xlim([0 20])
%axis([0 10 exactiters-1 2000 ])
if( write == 1 )
    saveas(h, plotnameiters, 'pdf');
end
hold off;
