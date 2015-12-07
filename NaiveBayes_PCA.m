function [yhat ,acc]=NaiveBayes_PCA(Xtrain, Xtest, Ytrain, Ytest)
nbpca = fitcnb(Xtrain, Ytrain);
yhat = predict(nbpca,Xtest);
acc = mean(Ytest==yhat);
end