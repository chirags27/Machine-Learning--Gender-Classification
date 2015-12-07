function avgL = averageLength(wordMatrix,words)
avg=zeros(1,size(wordMatrix,1));
for i=1:size(wordMatrix,1)
    for j=1:size(words,1)
    avg(i)= avg(i)+wordMatrix(i,j)*length(words{j});
    end;
    avg(i)=(avg(i)/sum(wordMatrix(i,:)));
end
avgL=avg;
end