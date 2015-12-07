wordfreq=importdata('words_train.txt');
gender=importdata('genders_train.txt');
images_transformed=importdata('image_features_train.txt');
image_data=importdata('images_train.txt');
%SVM With C=10 71.3 C=100 71.3 C=0.1 73.67 C=0.001 70.27
Xtrain=wordfreq(1:3999,:);
Xtest=wordfreq(4000:4998,:);
Ytrain=gender(1:3999);
Ytest=gender(4000:4998);


%Autoencoder
%acc=autoe(Xtrain);

%SVMMATLAB
[yhatsvm, accsvm]=svmtrainmatlab(Xtrain,Xtest,Ytrain,Ytest);

%SVM
%[yhatsvm, accsvm]=svmacc(Xtrain,Xtest,Ytrain,Ytest);

% 
% %NB on PCAed Data
%[coeff, score ,latent]=pca(wordfreq);
%figure,plot(cumsum(latent)/sum(latent));
%numpc=26;
%Xtrain_pca_w=score(1:3999,1:numpc);
%Xtest_pca_w=score(4000:4998,1:numpc);
%[yhat_nb, acc_nb]=NaiveBayes_PCA(Xtrain_pca_w, Xtest_pca_w ,Ytrain,Ytest);
% 
% 
% % Training Images NB % PCAed
%for i=1:(size(image_data,1))
%  cur_row=image_data(i,:);
%  cur_img=reshape(cur_row,[100 100 3]);
%  grayimg=rgb2gray(uint8(cur_img));
%  g(i,:)=reshape(grayimg,[1 10000]); 
%end;
%g=double(g);
% [coeff_i, latent_i]=pcacov((g)'*(g));
% figure,plot(cumsum(latent_i)/sum(latent_i));
% numpc_i=12;
%Xtrain_pca_wi=g(1:3999,:)*coeff_i(:,1:12);
%Xtest_pca_wi=g(4000:4998,:)*coeff_i(:,1:12);
% [yhat_nb_i, acc_nb_i]=NaiveBayes_PCA(Xtrain_pca_wi, Xtest_pca_wi ,Ytrain,Ytest);
% 
% 
% %Naive Bayes on transformed features
% Xtrain_pca_wtr=images_transformed(1:3999,:);
% Xtest_pca_wtr=images_transformed(4000:4998,:);
% [yhat_nb_itr, acc_nb_itr]=NaiveBayes_PCA(Xtrain_pca_wtr, Xtest_pca_wtr ,Ytrain,Ytest);
% 
% 
%%NaiveBayes For transformed + words
% 
 Xtrain_pca_trw=[images_transformed(1:3999,:) Xtrain_pca_w ];
 Xtest_pca_trw=[images_transformed(4000:4998,:) Xtest_pca_w ];
 [yhat_nb_itrw, acc_nb_itrw]=NaiveBayes_PCA(Xtrain_pca_trw, Xtest_pca_trw ,Ytrain,Ytest);
% 
% %Perceptron on the image data
% % %Initialize weights
% % gender1=gender;
% % gender(gender==0)=-1;
% % w = 0.0001*ones(10000,1);
% % %Keep a separate running sum of weights from each iteration
% % averaged_w = zeros(10000,1);
% % Rate=1;
% % for j=1:10
% % for i=1:(size(image_data,1)-1000)
% %   cur_row=image_data(i,:);
% %   cur_img=reshape(cur_row,[100 100 3]);
% %   grayimg=rgb2gray(uint8(cur_img));
% %   g=reshape(grayimg,[1 10000]);
% %   w= w + double(transpose(Rate*((gender(i)-sign(double(g)*w))*g));
% %   averaged_w=averaged_w+w;    
% % end;
% % end;
% % 
% % averaged_w=(averaged_w/(10*3999));
% % 
% % for i=4000:(size(image_data,1))
% %   cur_row=image_data(i,:);
% %   cur_img=reshape(cur_row,[100 100 3]);
% %   grayimg=rgb2gray(uint8(cur_img));
% %   g=reshape(grayimg,[1 10000]);
% %   pred(i-3999)=sign(double(g)*averaged_w);      
% % end;
% % 
% % 


%Decision Trees 53.85 with all the features(4000*10000)
%[yhat_dt acc_dt]=decisionTree(Xtrain_pca_trw,Xtest_pca_trw,Ytrain,Ytest);

%Logistic Regression
Ytrain_lr=Ytrain;
Ytrain_lr(Ytrain==0)=2;
%[yhat_lr acc_lr]=logistic_Regression(Xtrain_pca_trw,Xtest_pca_trw,Ytrain_lr,Ytest);
acc_lr_cv=cross_validation(@logistic_Regression,Xtrain_pca_trw,Ytrain_lr,4000);


%[yhatada accada]=Adaboost_Emsemble(Xtrain_pca_trw,Xtest_pca_trw,Ytrain,Ytest);
acc_ada_cv=cross_validation(@Adaboost_Emsemble,Xtrain_pca_trw,Ytrain,4000);


Yhat_temp_nb=yhat_nb_itrw;
Yhat_temp_nb(yhat_nb_itrw==0)=-1;
Yhat_temp_ada=yhatada;
Yhat_temp_ada(yhatada==0)=-1;
Yhat_temp_svm=yhatsvm;
Yhat_temp_svm(yhatsvm==0)=-1;
Yhat_temp_lr=yhat_lr;
Yhat_temp_lr(yhat_lr==0)=-1;


sumY_temp=sign(accada*Yhat_temp_ada+ acc_nb_itrw*Yhat_temp_nb+ (accsvm(1)/100)*Yhat_temp_svm ); %+ acc_lr*Yhat_temp_lr);

sumy=sumY_temp;
sumy(sumY_temp==-1)=0;

acc_final=mean(sumy==Ytest);