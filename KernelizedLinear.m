function [yhat_klr acc_klr] = KernelizedLinear(Xtrain, Xtest, Ytrain, Ytest)


yhat_klr=((Xtest*Xtrain').^2)*(inv(Xtrain*Xtrain' + 0.1*eye(size(Xtrain,1)))) * Ytrain;
hist(yhat_klr);
yhat_klr(yhat_klr>(-0.5))=0;
yhat_klr(yhat_klr<(-0.5))=1;
acc_klr=mean(yhat_klr==Ytest);

end

%[yhat_klr acc_klr]= cross_validation(@KernelizedLinear,images_transformed,Ytrain,4000);