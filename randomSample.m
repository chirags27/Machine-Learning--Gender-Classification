function [train_x,train_y,test_x,test_y] = randomSample(train_data,train_labels,prop_train)
    
    size_data = size(train_data);
    %size_train = (prop_train/(prop_test+prop_train))*size_data(1);
    sample_index = randperm(size_data(1),prop_train);
    
    train_x = train_data(sample_index,:);
    train_y = train_labels(sample_index,:);
    
    sample_index_test = setdiff(1:size_data(1),sample_index);
    
    test_x = train_data(sample_index_test,:);
    test_y = train_labels(sample_index_test,:);
    
end