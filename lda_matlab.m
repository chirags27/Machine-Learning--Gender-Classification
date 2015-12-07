function [yhat_lda acc_lda]=lda_matlab(Xtrain,Xtest,Ytrain,Ytest)

model=fitcdiscr((Xtrain),Ytrain);
yhat_lda=predict(model,(Xtest));
acc_lda=mean(yhat_lda==Ytest);

end