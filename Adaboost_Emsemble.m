function [yhat_ada acc_ada]=Adaboost_Emsemble(Xtrain,Xtest,Ytrain,Ytest)
 %ens=fitensemble(Xtrain,Ytrain,'Bag',500,'Tree',...
 %   'Type','Classification');
temp1= templateTree('MaxNumSplits',1);
ens = fitensemble(Xtrain,Ytrain,'RobustBoost',1000,temp1);
yhat_ada= predict(ens,Xtest);
acc_ada=mean(yhat_ada==Ytest);
end