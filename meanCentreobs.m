function X_mean=meanCentreobs(X)
X_mean=X-repmat(mean(X,2),[1 size(X,2)]);
end