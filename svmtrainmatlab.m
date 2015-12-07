function [yhatsvm_mat, accsvm_mat]=svmtrainmatlab(Xtrain,Xtest,Ytrain,Ytest)
SVMModel = fitcsvm(Xtrain,Ytrain,'KernelFunction','kernel_intersection');
yhatsvm_mat=predict(SVMModel,Xtest);
accsvm_mat=mean(yhatsvm_mat==Ytest);
end