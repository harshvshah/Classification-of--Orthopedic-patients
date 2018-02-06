data2 = readtable("Biomechanical_Data_column_2C_weka_q3.csv");

[train_ind, val_ind_1, test_ind] = dividerand(310,0.6774,0.0,0.3226);

train_data2_1 = data2(train_ind, :);
test_data2_1 = data2(test_ind, :);

[train_ind, val_ind_2, test_ind] = dividerand(310,0.6774,0.0,0.3226);

train_data2_2 = data2(train_ind, :);
test_data2_2 = data2(test_ind, :);

[train_ind, val_ind_3, test_ind] = dividerand(310,0.6774,0.0,0.3226);

train_data2_3 = data2(train_ind, :);
test_data2_3 = data2(test_ind, :);
[train_ind, val_ind_4, test_ind] = dividerand(310,0.6774,0.0,0.3226);

train_data2_4 = data2(train_ind, :);
test_data2_4 = data2(test_ind, :);

[train_ind, val_ind_5, test_ind] = dividerand(310,0.6774,0.0,0.3226);

train_data2_5 = data2(train_ind, :);
test_data2_5 = data2(test_ind, :);

decisiontree_data2_10_1 = fitctree(train_data2_1, 'class', 'MinLeafSize', 10);
decisiontree_data2_10_2 = fitctree(train_data2_2, 'class', 'MinLeafSize', 10);
decisiontree_data2_10_3 = fitctree(train_data2_3, 'class', 'MinLeafSize', 10);
decisiontree_data2_10_4 = fitctree(train_data2_4, 'class', 'MinLeafSize', 10);
decisiontree_data2_10_5 = fitctree(train_data2_5, 'class', 'MinLeafSize', 10);

predicted_data2_10_1 = predict(decisiontree_data2_10_1, test_data2_1);
predicted_data2_10_2 = predict(decisiontree_data2_10_2, test_data2_2);
predicted_data2_10_3 = predict(decisiontree_data2_10_3, test_data2_3);
predicted_data2_10_4 = predict(decisiontree_data2_10_4, test_data2_4);
predicted_data2_10_5 = predict(decisiontree_data2_10_5, test_data2_5);

actual_data2_1 = table2array(test_data2_1(:, 7));
actual_data2_2 = table2array(test_data2_2(:, 7));
actual_data2_3 = table2array(test_data2_3(:, 7));
actual_data2_4 = table2array(test_data2_4(:, 7));
actual_data2_5 = table2array(test_data2_5(:, 7));

actual_data2_norm_1(strcmp(actual_data2_1, 'Abnormal')) = 0;
actual_data2_norm_1(strcmp(actual_data2_1, 'Normal')) = 1;
actual_data2_norm_2(strcmp(actual_data2_2, 'Abnormal')) = 0;
actual_data2_norm_2(strcmp(actual_data2_2, 'Normal')) = 1;
actual_data2_norm_3(strcmp(actual_data2_3, 'Abnormal')) = 0;
actual_data2_norm_3(strcmp(actual_data2_3, 'Normal')) = 1;
actual_data2_norm_4(strcmp(actual_data2_4, 'Abnormal')) = 0;
actual_data2_norm_4(strcmp(actual_data2_4, 'Normal')) = 1;
actual_data2_norm_5(strcmp(actual_data2_5, 'Abnormal')) = 0;
actual_data2_norm_5(strcmp(actual_data2_5, 'Normal')) = 1;

predicted_data2_10_norm_1(strcmp(predicted_data2_10_1, 'Abnormal')) = 0;
predicted_data2_10_norm_1(strcmp(predicted_data2_10_1, 'Normal')) = 1;
predicted_data2_10_norm_2(strcmp(predicted_data2_10_2, 'Abnormal')) = 0;
predicted_data2_10_norm_2(strcmp(predicted_data2_10_2, 'Normal')) = 1;
predicted_data2_10_norm_3(strcmp(predicted_data2_10_3, 'Abnormal')) = 0;
predicted_data2_10_norm_3(strcmp(predicted_data2_10_3, 'Normal')) = 1;
predicted_data2_10_norm_4(strcmp(predicted_data2_10_4, 'Abnormal')) = 0;
predicted_data2_10_norm_4(strcmp(predicted_data2_10_4, 'Normal')) = 1;
predicted_data2_10_norm_5(strcmp(predicted_data2_10_5, 'Abnormal')) = 0;
predicted_data2_10_norm_5(strcmp(predicted_data2_10_5, 'Normal')) = 1;

plotconfusion(actual_data2_norm_1, predicted_data2_10_norm_1);
plotconfusion(actual_data2_norm_2, predicted_data2_10_norm_2);
plotconfusion(actual_data2_norm_3, predicted_data2_10_norm_3);
plotconfusion(actual_data2_norm_4, predicted_data2_10_norm_4);
plotconfusion(actual_data2_norm_5, predicted_data2_10_norm_5);
