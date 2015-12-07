
yhat = sign(Xtest*m_sf);
acc=mean(yhat==Ytest);
