% Results-- mean centering accross rows doesn't make a difference
wordfreq=importdata('words_train.txt');
m=mean(wordfreq,2);
meanrowMat=repmat(m,[1 size(wordfreq,2)]);
wordfreq_meanedRow=wordfreq-meanrowMat; % subtracting mean across all the rows
wordfreq_meanedRow(wordfreq==0)=0; % Removing elements which were already negative
wordfreq_meanedRow
[coeff score latent]=pca(wordfreq_meanedRow); % 
figure, plot(cumsum(latent)/sum(latent));