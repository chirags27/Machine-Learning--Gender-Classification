function K = kernel_gaussian(X, X2, sigma)
% Evaluates the Gaussian Kernel with specified sigma
%
% Usage:
%
%    K = KERNEL_GAUSSIAN(X, X2, SIGMA)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the Guassian kernel
% with parameter sigma=20.
if(nargin==2)
    sigma=20;
end;
    
n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.

X_transpose=transpose(X);
X2_transpose=transpose(X2);

for i=1:m
    for j=1:n
K(i,j) = exp(-sum((X2_transpose(:,i)-X_transpose(:,j)).^2)/(2*sigma^2));
    end;
end;
