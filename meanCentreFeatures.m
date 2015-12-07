function X_mean= meanCentreFeatures(X)
X_mean=X-repmat(mean(X,1),[size(X,1) 1]);
end