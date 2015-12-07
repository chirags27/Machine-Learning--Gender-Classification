function K = kernel_probability(X, X2)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    K = KERNEL_INTERSECTION(X, X2)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the histogram
% intersection kernel.
X=X./repmat(sum(X,2),[1 size(X,2)]);
X2=X2./repmat(sum(X2,2),[1 size(X2,2)]);

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
        
        temp= sum(abs(X2_transpose(:,i)-X_transpose(:,j)));
        if(temp<0.5)
        K(i,j)=1 ;
        else
         K(i,j)=0 ;
        end;
    end;
end;
