load('trained_networks_and_data.mat', 'nn1', 'Snn', 'test_x', 'test_y');

% Compute and plot the correlations
[Cbefores, Cafters] = compute_activation_correlation_2tasks(nn1,Snn,test_x,test_y);

disp(['Correlations before SRC: ' Cbefores]);
disp(['Correlations after SRC: ' Cafters]);