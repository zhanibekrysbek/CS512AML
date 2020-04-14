x = textread('monitor.txt');
x(1,:) = [];

figure;
plot(x(:,1), x(:,2), 'r');
xlabel('Iteration');
ylabel('Objective value');

figure
plot(x(:,1), 100-x(:,end), 'k');
xlabel('Iteration');
ylabel('Letter-wise accuracy');
