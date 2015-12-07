function [yhat_linearreg acc_linearreg]= LinearReg(Xtrain,Xtest,Ytrain,Ytest)
B  =fitlm(Xtrain,Ytrain);
yhat_linearreg= predict(B,Xtest);
yhat_linearreg(yhat_linearreg>=0.5)=1;
yhat_linearreg(yhat_linearreg<0.5)=0;
acc_linearreg=mean(yhat_linearreg==Ytest);
end