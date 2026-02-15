load('Figure5.mat','HL1','HL2');

% Plot Figure 5
% Plot the results for Task 1 and Task 2
figure;

% Hidden Layer 1
subplot(1, 2, 1);
scatter(HL1{1},HL1{2},50,"MarkerEdgeColor","red") % Task 1
hold on; % Ensure that the next scatter plot is added to the same figure
scatter(HL1{3},HL1{4},50,"MarkerEdgeColor","blue") % Task 2
hold on;
plot(linspace(-0.6,0,100), linspace(-0.6,0,100), 'k--', 'LineWidth', 0.8); % Plot the line y = x in dashed black
hold off;
title('Hidden Layer 1');
xlabel('Pre sleep input');
ylabel('Post sleep input');
legend('Task 1','Task 2','y = x','Location','northwest');

% Hidden Layer 2
subplot(1, 2, 2);
scatter(HL2{1},HL2{2},50,"MarkerEdgeColor","red") % Task 1
hold on; % Ensure that the next scatter plot is added to the same figure
scatter(HL2{3},HL2{4},50,"MarkerEdgeColor","blue") % Task 2
hold on;
plot(linspace(-0.0015,0.0015,100), linspace(-0.0015,0.0015,100), 'k--', 'LineWidth', 0.8); % Plot the line y = x in dashed black
hold off;
title('Hidden Layer 2');
xlabel('Pre sleep input');
ylabel('Post sleep input');
legend('Task 1','Task 2','y = x','Location','northwest');