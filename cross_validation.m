function [mean_accuracy] = cross_validation(method_handle, train_data, train_labels, prop_train)
    %prop_train=(4*prop_train/5);
    max_iter = 10;
    mean_accuracy =0;
    for i = 1:max_iter
        [train_x_sample,train_y_sample,test_x_sample,test_y_sample] = randomSample(train_data,train_labels,prop_train);
        predicted_labels = method_handle(train_x_sample,test_x_sample,train_y_sample,test_y_sample);
        
        %sum(predicted_labels==1);
        
        %sum(test_y_sample==2);
        
        %predicted_labels(predicted_labels==0)=2;
        %sum(predicted_labels)
        
        
        
        accuracy = mean(predicted_labels == test_y_sample);
        disp('Iter');
        disp(accuracy);
        mean_accuracy = mean_accuracy + accuracy;
    end
    
    mean_accuracy = mean_accuracy/max_iter
    
    %[~, best_val_at]=min(accuracy);
end