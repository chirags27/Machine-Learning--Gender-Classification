%{

    words_train is very sparse. 64% is empty(0 values). 25 PC's explain 90%

    of the variance.

%}
wordfreq=importdata('words_train.txt');
genderM=dlmread('genders_train.txt');
%[Coefficients, Scores, latent] = pca(wordfreq);
%wordfreq=Scores;
%wordfreq=wordfreq(:,1:26);
wordfreq1=wordfreq(4000:4998,:);
wordfreq=wordfreq(1:3999,:);
%wordfreq=kernel_poly(wordfreq,wordfreq,2);
%wordfreq1=kernel_poly(wordfreq1,wordfreq1,2);
model = svmtrain(genderM(1:3999,:), [(1:size(wordfreq,1))' wordfreq], sprintf('-t 1 -c %g', 10));
%For the test data
[yhat acc vals] = svmpredict(genderM(4000:4998,:),[(1:size(wordfreq1,1))' wordfreq1], model);
%Error on train data
%[yhat acc vals] = svmpredict(genderM, [(1:size(wordfreq,1))' wordfreq], model);
test_err = mean(yhat~=genderM(4000:4998,:));

%size(Scores)

%class_male = Scores(find(genders_train == 0),:);

%class_female = Scores(find(genders_train == 1),:);

 

 

%size(class_male)

%size(class_female)

 

%figure

%plot(1:numel(latent),cumsum(latent)/sum(latent));

%figure

%plot(class_male(:,1),class_male(:,2),'+');

%hold on

%plot(class_female(:,1),class_female(:,2),'o');

 

%legend('Male','Female')

%xlabel('PCA1')

%ylabel('PCA2')