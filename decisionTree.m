function [yhat acc]= decisionTree(Xtrain,Xtest,Ytrain,Ytest)
tree = fitctree(Xtrain,Ytrain);
yhat=predict(tree,Xtest);
acc=mean(yhat==Ytest);
end