function [yhat_stw acc_stw]=stepwise(Xtrain,Xtest,Ytrain,Ytest)
b = stepwisefit(Xtrain,Ytrain);
yhat_stw=sign(Xtest*b);
%figure,histfit(yhat_stw);
yhat_stw=sign(Xtest*b);
yhat_stw(yhat_stw==-1)=0;
%figure,histfit(yhat_stw);
%yhat_stw(yhat_stw<0.5)=0;
acc_stw=mean(yhat_stw==Ytest);
end