%Feature 3 is for specs. 
% 931, 1000's! Not really a differentiating factor
%Mean age of females is 25, while males is 33
% On an average, women smile more-around 59.91 and males 43.47
% On an average, the face size is same! 0.13's
wordfreq=importdata('words_train.txt');
gender=importdata('genders_train.txt');
images_transformed=importdata('image_features_train.txt');
image_data=importdata('images_train.txt');
Xtest_words=importdata('words_test.txt');
Xtest_imgfeatures=importdata('image_features_test.txt');

%SVM With C=10 71.3 C=100 71.3 C=0.1 73.67 C=0.001 70.27
Ytrain=gender;
Ytest=zeros(size(Xtest_words,1),1);
%[yhatsvm, accsvm]=svmacc(wordfreq,Xtest_words,Ytrain,Ytest);


%SVM on MATLAB
%Only on words
%SVMMATLAB
[yhatsvm, accsvm]=svmtrainmatlab(wordfreq,Xtest_words,Ytrain,Ytest);
%[accsvm_cv]=cross_validation(@svmtrainmatlab,wordfreq,Ytrain,4000);

%PCA -> NB and Adaboost
[coeff_w, score_w, latent_w]=pca([wordfreq;Xtest_words]);
%figure,plot(cumsum(latent_w)/sum(latent_w));

% Considering 94 % recons numpc=90
Xtrain_w_pca=score_w(1:size(wordfreq,1),1:100);
Xtest_w_pca=score_w(size(wordfreq,1)+1:size(score_w),1:100);

Xtrain_trw_pca=[Xtrain_w_pca images_transformed];
Xtest_trw_pca=[Xtest_w_pca Xtest_imgfeatures];

%SVM on words and image_trans
%[yhatsvm, accsvm]=svmtrainmatlab(Xtrain_trw_pca,Xtest_trw_pca,Ytrain,Ytest);

%Stepwise

%acc_stw_cv=cross_validation(@stepwise,Xtrain_w_pca,Ytrain,4000);
%Mean centered gives better stuff
acc_stw_cv=cross_validation(@stepwise,meanCentreFeatures(Xtrain_w_pca),Ytrain,4000);
[yhat_stw, acc_stw]=stepwise(Xtrain_w_pca,Xtest_w_pca,Ytrain,Ytest);


%NB
%[yhat_nb_itrw, acc_nb_itrw]=NaiveBayes_PCA(Xtrain_trw_pca, Xtest_trw_pca ,Ytrain,Ytest);
%[acc_nb_cv]=cross_validation(@NaiveBayes_PCA,Xtrain_trw_pca,Ytrain,4000);

%Adaboost
[yhatada, accada]=Adaboost_Emsemble(Xtrain_trw_pca,Xtest_trw_pca,Ytrain,Ytest);
%[acc_ada_cv]=cross_validation(@Adaboost_Emsemble,Xtrain_trw_pca,Ytrain,4000);

%Linear Lasso.
[yhat_linearreg, acc_linearreg]=LinearReg(Xtrain_trw_pca,Xtest_trw_pca,Ytrain,Ytest);
%Linear CV:
%[acc_linr_cv]=cross_validation(@LinearReg,Xtrain_trw_pca,Ytrain,4000);


%Logistic Regression

[yhat_lr, acc_lr]=logistic_Regression(Xtrain_trw_pca,Xtest_trw_pca,Ytrain,Ytest);
%[acc_lr_cv]=cross_validation(@logistic_Regression,Xtrain_trw_pca,Ytrain,4000);
% Only consider features im -> 1 2 and 100 words pca -> ACC 84.67

%Sum of Stuff
%Yhat_temp_nb=yhat_nb_itrw;
%Yhat_temp_nb(yhat_nb_itrw==0)=-1;

%% Train models
Yhat_temp_ada=yhat_ada;
Yhat_temp_ada(yhat_ada==0)=-1;

Yhat_temp_lda=yhat_lda;
Yhat_temp_lda(yhat_lda==0)=-1;

Yhat_temp_svm=yhatsvm;
Yhat_temp_svm(yhatsvm==0)=-1;

Yhat_temp_svm_wi=yhatsvm_wi;
Yhat_temp_svm_wi(yhatsvm_wi==0)=-1;

%
%% Test Models
Yhat_temp_svm_=yhatsvm1t_;
Yhat_temp_svm_(yhatsvm1t_==0)=-1;

Yhat_temp_svm_wi_=yhatsvm2t_;
Yhat_temp_svm_wi_(yhatsvm2t_==0)=-1;

Yhat_temp_linr_=yhatlinrt_;
Yhat_temp_linr_(yhatlinrt_==0)=-1;

Yhat_temp_lda_=yhatldat_;
Yhat_temp_lda_(yhatldat_==0)=-1;
%Test models end
%% 

sumY_temp=(Yhat_temp_ada + Yhat_temp_svm + Yhat_temp_svm_ + Yhat_temp_svm_wi + 2*Yhat_temp_svm_wi_ +  Yhat_temp_linr_ + Yhat_temp_lda_+ Yhat_temp_lda);
% count=0;
% for i=1:size(Yhat_temp_ada,1)
% if(sumY_temp(i,1)==1 || sumY_temp(i,1)==-1)
% sumY_temp(i,1)=Yhat_temp_svm(i,1);
%     count=count+1;
% end
% end;

%count
%count=0;
sumY_temp=sign(sumY_temp);
sumy=sumY_temp;
sumy(sumY_temp==-1)=0;

acc_final=mean(sumy==Ytest);  %Accuracy 


%[coeff_tri score_tri latent_tri]=pca([images_transformed;Xtest_imgfeatures]);

%Xtrain_impca=score_tri(1:size(images_transformed,1),1:5);
%Xtest_impca=score_tri(1:size(Xtest_imgfeatures,1),1:5);

count=0;
submission=yhatsvm;
for i=1:size(submission,1)
    if(yhatsvm(i,1)~= yhat_ada(i,1) &&yhat_linearreg(i,1)==yhat_ada(i,1) && yhat_stw(i,1)==yhat_ada(i,1) && yhat_ada(i,1) == yhatsvm_wi(i,1))
        submission(i,1)=yhat_ada(i,1);
        count=count+1;
        count
    end;
end;

count=1;
for i=1:size(yhatada,1)
if((yhatsvm(i,1)== yhatsvm_wi(i,1))) 
idx_(count,1)=i;
count=count+1;
end;
end;
count

% [pca_normalized_wco pca_normalized_w_score pca_normalized_w_latent] = pca([impwords_w_train;impwords_w_test]);

plot(cumsum(pca_normalized_w_latent)/sum(pca_normalized_w_latent))

        Xtrain_npca=[pca_normalized_w_score(1:4998,1:420) images_transformed(:,[1 2 7])];
         Xtrain_npca=[pca_normalized_w_score(1:4998,1:420) images_transformed(:,[1 2 7])];
Xtest_npca=[pca_normalized_w_score(4999:9995,1:420) Xtest_imgfeatures(:,[1 2 7])];




%%
%Final 

idx_ = zeros(4997,1);
c1=1;
c2=2;
c3=3;

y = (yhatsvm + yhatsvm_wi + yhat_ada + yhatldat_ + yhatsvm2t_ + yhatlinrt_ + yhat_lda)/7;
for i=1:size(y)
if(y(i,1)>=0.5)
    y(i,1)=1;
else
    y(i,1)=0;
end;
end;

submission= y;
for i=1:size(yhatsvm,1)
    % ALL
    if(yhatsvm(i,1)== yhatsvm_wi(i,1) && (yhatsvm(i,1)== yhatsvm1t_(i,1)) && yhatsvm(i,1)== yhatsvm2t_(i,1)&& yhatsvm(i,1)== yhatldat_(i,1) && yhatsvm(i,1)== yhatlinrt_(i,1) && yhatsvm(i,1)== yhat_ada(i,1) && yhatsvm(i,1)== yhat_lda(i,1))
        submission(i,1)=yhatsvm(i,1);
        idx_(c1,1) = i; 
        c1=c1+1;
    % Train
    elseif(yhatsvm(i,1)== yhatsvm_wi(i,1) && yhatsvm(i,1)== yhat_ada(i,1) && yhatsvm(i,1)== yhat_lda(i,1))
        submission(i,1) = yhatsvm(i,1);
          c2=c2+1;
    % Test
    elseif(yhatsvm1t_(i,1)== yhatsvm2t_(i,1) && yhatsvm1t_(i,1)== yhatlinrt_(i,1) && yhatsvm1t_(i,1)== yhatldat_(i,1))
        submission(i,1) = yhatsvm1t_(i,1);
        c3=c3+1;
    end;
end;

c1
c2
c3

%%
% count = 0;
% for i=1:size(y1)
%     if(y1(i,1) == y2(i,1) && y3(i,1) == y2(i,1) && y4(i,1) == y2(i,1))
%         count=count+1;
%     end;
% end;
% count