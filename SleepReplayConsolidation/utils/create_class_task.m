function [task_assignment, newY] = create_class_task(X,y)
%creates the class task in the Gido paper: 
% 5 tasks representing digits 0-1, 2-3, 4-5, 6-7, 8-9
% and the label says which digit it is

num_images = size(X,1);

task_assignment = zeros(num_images,1);
newY = y;

[~, labels] = max(y, [], 2); % non ho capito perché ma in questo progetto train_y del MNIST dataset ha dimensioni (60000,10) invece che (60000,) e quindi per decidere il label dell'immagine loro prendono il massimo dei 10 valori
% Callo è un genio è mi ha giustamente fatto notare che per ogni immagine
% (quindi riga) tutte le cifre sono 0 tranne una che è 1, quindi stanno
% codificando i valori 0-9 con 9 zeri e 1 uno, i.e. 5 = 0000100000

for i = 1:num_images
    if labels(i) == 1 || labels(i) == 2
        task_assignment(i) = 1;
    elseif labels(i) == 3 || labels(i) == 4
        task_assignment(i) = 2;
    elseif labels(i) == 5 || labels(i) == 6
        task_assignment(i) = 3;
    elseif labels(i) == 7 || labels(i) == 8
        task_assignment(i) = 4;
    elseif labels(i) == 9 || labels(i) == 10
        task_assignment(i) = 5;
    end
end
