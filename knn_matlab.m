function [yhat_knn acc_knn]= knn_matlab(Xtrain,Xtest,Ytrain,Ytest)
%temp=(1/(@kernel_intersection));

md = fitcknn(Xtrain,Ytrain,'Distance','minkowski','DistanceWeight','inverse');
yhat_knn=predict(md,Xtest);
acc_knn=mean(yhat_knn==Ytest);
end