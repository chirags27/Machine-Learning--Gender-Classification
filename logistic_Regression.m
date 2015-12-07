function [yhat_final acc]= logistic_Regression(Xtrain,Xtest,Ytrain,Ytest)

Ytrain(Ytrain==0)=2;

B = mnrfit(Xtrain,Ytrain);
yhat=mnrval(B,Xtest);
yhat(yhat==2)=0;
max(yhat);
size(Ytest);

temp=(yhat>0.5);
yhat_final=temp(:,1);
%Add this line for For CV
yhat_final=double(yhat_final);
%yhat_final(yhat_final==0)=2;
%
%yhat_final
acc=mean(yhat_final==Ytest);
end