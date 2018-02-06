data3 = readtable("BiomechanicalData_column_3C_weka.csv");

[train_ind, val_ind, test_ind] = dividerand(310,0.6774,0.0,0.3226);

train_data3 = data3(train_ind, :);
test_data3 = data3(test_ind, :);

decisiontree_data3 = fitctree(train_data3, 'class', 'MinLeafSize', 10);
 view(decisiontree_data3, 'Mode', 'graph');

predicted_data3 = predict(decisiontree_data3, test_data3);
actual_data3 = table2array(test_data3(:, 7));

actual_data3_norm(strcmp(actual_data3, 'Hernia')) = 1;
actual_data3_norm(strcmp(actual_data3, 'Spondylolisthesis')) = 2;
actual_data3_norm(strcmp(actual_data3, 'Normal')) = 3;
predicted_data3_norm(strcmp(predicted_data3, 'Hernia')) = 1;
predicted_data3_norm(strcmp(predicted_data3, 'Spondylolisthesis')) = 2;
predicted_data3_norm(strcmp(predicted_data3, 'Normal')) = 3;

% Convert this data to a [3 x 100] matrix
targets = zeros(3,100);
outputs = zeros(3,100);
targetsIdx = sub2ind(size(targets), actual_data3_norm, 1:100);
outputsIdx = sub2ind(size(outputs), predicted_data3_norm, 1:100);
targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;

plotconfusion(targets, outputs);
