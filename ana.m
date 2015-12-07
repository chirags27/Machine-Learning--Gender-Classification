wordfreq1 = bsxfun(@rdivide,wordfreq,sum(wordfreq.^2,2).^0.5);
genderMaleWords=wordfreq1(gender==0,:);
genderFeMaleWords=wordfreq1(gender==1,:);
genderMaleWords=mean(genderMaleWords);
genderFeMaleWords=mean(genderFeMaleWords);
a=(abs(genderMaleWords - genderFeMaleWords));
b=sort(a,'descend');
[Lia, posi]= ismember(b(1,1:1500),a);
impwords_w_train=wordfreq1(:,posi);

wordfreq1_test = bsxfun(@rdivide,Xtest_words,sum(Xtest_words.^2,2).^0.5);
impwords_w_test=wordfreq1_test(:,posi);
%[acc_nb_cv]=cross_validation(@NaiveBayes_PCA,impwords_w,Ytrain,4000);
%impwords=(find(a,b(1,1:100)));

%[acc_linr_cv]=cross_validation(@LinearReg,impwords_w,Ytrain,4000);

%
%[acc_ada_cv]=cross_validation(@Adaboost_Emsemble,impwords_w,Ytrain,4000);
% 88ish with 1500 withut mean centering
[accada_cv_im_h]=cross_validation(@Adaboost_Emsemble,[impwords_w_train (images_transformed)],Ytrain,4000);
[yhat_ada, acc_ada]=Adaboost_Emsemble([impwords_w_train(:,:) (images_transformed(:,:))],[impwords_w_test(:,:) (Xtest_imgfeatures(:,:))],Ytrain,Ytest);

%[acc_ada_lr]=cross_validation(@logistic_Regression,impwords_w,Ytrain,4000);

% acclda_cv = 87ish
%[acclda_cv]=cross_validation(@lda_matlab,[meanCentreFeatures(impwords_w_train(:,1:500)) meanCentreFeatures(images_transformed(:,[1 2]))],Ytrain,4000);
[yhat_lda, acc_lda]=lda_matlab([meanCentreFeatures(impwords_w_train(:,1:500)) meanCentreFeatures(images_transformed(:,[1 2]))],[meanCentreFeatures(impwords_w_test(:,1:500)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain],Ytest);


%Around 85.87 for n=1500
%acc_stw_cv=cross_validation(@stepwise,[meanCentreFeatures(impwords_w) meanCentreFeatures(images_transformed)],Ytrain,4000);
[yhat_stw, acc_stw]=stepwise([meanCentreFeatures(impwords_w_train) meanCentreFeatures(images_transformed)],[meanCentreFeatures(impwords_w_test) meanCentreFeatures(Xtest_imgfeatures)],Ytrain,Ytest);
  
  
%87.80 with 2 image transformed features 
%[accsvm_cv_wi]=cross_validation(@svmtrainmatlab,[(impwords_w_train) meanCentreFeatures(images_transformed(:,[1 2])],Ytrain,4000);
[yhatsvm_wi, accsvm]=svmtrainmatlab([meanCentreFeatures(impwords_w_train(:,:)) meanCentreFeatures(images_transformed(:,[1 2]))],[meanCentreFeatures(impwords_w_test(:,:)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],Ytrain,Ytest);



%88.78 without meanCentering the words
%[accsvm_cv]=cross_validation(@svmtrainmatlab,[(impwords_w)],Ytrain,4000);
%% Train models
[yhatsvm, accsvm]=svmtrainmatlab(impwords_w_train(:,1:600),impwords_w_test(:,1:600),Ytrain,Ytest);

[yhatsvm_wi, accsvm]=svmtrainmatlab([meanCentreFeatures(impwords_w_train(:,1:600)) meanCentreFeatures(images_transformed(:,[1 2]))],[meanCentreFeatures(impwords_w_test(:,1:600)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],Ytrain,Ytest);

[yhat_lda, acc_lda]=lda_matlab([meanCentreFeatures(impwords_w_train(:,1:500)) meanCentreFeatures(images_transformed(:,[1 2]))],[meanCentreFeatures(impwords_w_test(:,1:500)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain],Ytest);

[yhat_ada, acc_ada]=Adaboost_Emsemble([impwords_w_train(:,1:600) (images_transformed(:,1:2))],[impwords_w_test(:,1:600) (Xtest_imgfeatures(:,1:2))],Ytrain,Ytest);

%[yhat_lr, acc_lr]=logistic_Regression([meanCentreFeatures(impwords_w_train(:,1:300)) meanCentreFeatures(images_transformed(:,[1 2]))],[meanCentreFeatures(impwords_w_test(:,1:300)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],Ytrain,Ytest);
%[yhat_linearreg, acc_linr]=LinearReg([meanCentreFeatures(impwords_w_train(:,:)) meanCentreFeatures(images_transformed(:,[1 2]))],[meanCentreFeatures(impwords_w_test(:,:)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],Ytrain,Ytest);
%[yhat_linearreg, acc_linr]=LinearReg([(impwords_w_train(:,:)) (images_transformed(:,[1 2]))],[(impwords_w_test(:,:)) (Xtest_imgfeatures(:,[1 2]))],Ytrain,Ytest);


%%
%knn
acc_knn_cv=cross_validation(@knn_matlab,impwords_w_train,Ytrain,4000);


%All models final 800 - > 93ish
[yhatsvm_, accsvm]=svmtrainmatlab([impwords_w_train(:,1:800) ; impwords_w_test(idx_,1:800)],impwords_w_test(:,1:800),[Ytrain; Y_best(idx_,1)],Ytest);

%use 800
[yhatsvm_wi_, accsvm]=svmtrainmatlab([meanCentreFeatures([impwords_w_train(:,1:800) ; impwords_w_test(idx_,1:800)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:800)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

%use 800
[yhat_ada_, acc_ada]=Adaboost_Emsemble([([impwords_w_train(:,1:800) ; impwords_w_test(idx_,1:800)]) ([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[(impwords_w_test(:,1:800)) (Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

%use 800
[yhat_stw_, acc_stw]=stepwise([meanCentreFeatures([impwords_w_train(:,1:800) ; impwords_w_test(idx_,1:800)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:800)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

%use 800
[yhat_linearreg_, acc_linr]=LinearReg([meanCentreFeatures([impwords_w_train(:,1:800) ; impwords_w_test(idx_,1:800)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:800)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

%use 300
[yhat_lr, acc_lr]=logistic_Regression([meanCentreFeatures([impwords_w_train(:,1:300) ; impwords_w_test(idx_,1:300)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:300)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);



%use 800
[yhat_lda, acc_lda]=lda_matlab([meanCentreFeatures([impwords_w_train(:,1:800) ; impwords_w_test(idx_,1:800)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:800)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);






%% Test Models
[yhatlinrt_, acc]=LinearReg(([impwords_w_test(idx_,1:600) Xtest_imgfeatures(idx_,1:2)]),([impwords_w_test(:,1:600) Xtest_imgfeatures(:,1:2)]),Y_best(idx_,1),Ytest);

[yhatsvm1t_, acc]=svmtrainmatlab(([impwords_w_test(idx_,1:600) ]),([impwords_w_test(:,1:600)]),Y_best(idx_,1),Ytest);

[yhatsvm2t_, acc]=svmtrainmatlab(meanCentreFeatures([impwords_w_test(idx_,1:600) Xtest_imgfeatures(idx_,1:2)]),meanCentreFeatures([impwords_w_test(:,1:600) Xtest_imgfeatures(:,1:2)]),Y_best(idx_,1),Ytest);

[yhatldat_, acc]=lda_matlab(meanCentreFeatures([impwords_w_test(idx_,1:600) Xtest_imgfeatures(idx_,1:2)]),meanCentreFeatures([impwords_w_test(:,1:600) Xtest_imgfeatures(:,1:2)]),Y_best(idx_,1),Ytest);

%% Train + Test



% 90.9 with 300 words logistic
%cross_validation(@logistic_Regression,[meanCentreFeatures([impwords_w_train(:,1:300) ; impwords_w_test(idx_,1:300)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[Ytrain; Y_best(idx_,1)],7000);

%92 - 93 ish with 800 Linear
%cross_validation(@LinearReg,[meanCentreFeatures([impwords_w_train(:,1:800) ; impwords_w_test(idx_,1:800)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[Ytrain; Y_best(idx_,1)],7000);

[yhat_lda_tt, acc_lda]=lda_matlab([meanCentreFeatures([impwords_w_train(:,1:600) ; impwords_w_test(idx_,1:600)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:600)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

[yhat_svm1_tt, acc_svm2]=svmtrainmatlab(([impwords_w_train(:,1:600) ; impwords_w_test(idx_,1:600)]),(impwords_w_test(:,1:600)),[Ytrain;Y_best(idx_,1)],Ytest);

[yhat_svm2_tt, acc_svm1]=svmtrainmatlab([meanCentreFeatures([impwords_w_train(:,1:600) ; impwords_w_test(idx_,1:600)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:600)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

[yhat_ada_tt, acc_ada]=Adaboost_Emsemble([([impwords_w_train(:,1:600) ; impwords_w_test(idx_,1:600)]) ([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[(impwords_w_test(:,1:600)) (Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

[yhat_linr_tt, acc_linr]=LinearReg([([impwords_w_train(:,1:600) ; impwords_w_test(idx_,1:600)]) ([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[(impwords_w_test(:,1:600)) (Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

[yhat_stw_tt, acc_stw]=stepwise([meanCentreFeatures([impwords_w_train(:,1:600) ; impwords_w_test(idx_,1:600)]) meanCentreFeatures([images_transformed(:,[1 2]); Xtest_imgfeatures(idx_,[1 2])])],[meanCentreFeatures(impwords_w_test(:,1:600)) meanCentreFeatures(Xtest_imgfeatures(:,[1 2]))],[Ytrain;Y_best(idx_,1)],Ytest);

[yhat_lr_tt, acc_lr]=logistic_Regression([([impwords_w_train(:,1:200) ; impwords_w_test(idx_,1:200)]) ([images_transformed(:,:); Xtest_imgfeatures(idx_,:)])],[(impwords_w_test(:,1:600)) (Xtest_imgfeatures(:,:))],[Ytrain;Y_best(idx_,1)],Ytest);

%mean(genderMaleWords)
%mean(genderFeMaleWords)
%men=(genderMaleWords==0);
%women = (genderFeMaleWords==0);
%genderMaleWords=(genderMaleWords<10);
% genderFeMaleWords=(genderFeMaleWords>1000);
 
%analysis=sum(genderMaleWords==genderFeMaleWords);
% sparse(genderMaleWords==genderFeMaleWords)


% Playing with images.
matrix=[-1 -1 -1;-1 8 -1; -1 -1 -1 ];
for i=1:size(g,1)
    temp_im=uint8(reshape(g(i,:),[100 100]));
end

temp_edge=temp_im;
for i=2:size(temp_im,1)-2
    for j=2: size(temp_im,2)-2
    temp_edge(i,j) = mean(mean(double(temp_im(i:i+2,j:j+2)).*matrix));
    end
end


immale=g(gender==0,:);
imfemale=g(gender==1,:);


%for i=1:size(Xtest,1)
%c(i,1)=fisherfaces_predict(m_ff,Xtest(i,:)',10);
%end;

for i=1:size(g,1)
c2(i,1)=eigenfaces_predict(m_ef,g(1000,:)',5)
end;



male_im=mean(g(gender==0,:));
female_im=mean(g(gender==1,:));
temp_im_male=uint8(reshape(male_im,[100 100]));
temp_im_female=uint8(reshape(female_im,[100 100]));

for i=1:size(g,1)
if(sum(g(i,:)-male_im).^2 < sum(g(i,:)-female_im).^2)
soln(i,1)=0;
else
soln(i,1)=1;
end;
end


% [c_i s_i l_i]=pcacov(g'*g);
% male_im=mean(images_transformed(gender==0,:));
% female_im=mean(images_transformed(gender==1,:));
% 
% for i=1:size(g,1)
% if(sum(images_transformed(i,:)-male_im).^2 < sum(images_transformed(i,:)-female_im).^2)
% soln(i,1)=0;
% else
% soln(i,1)=1;
% end;
% end


% male_im=mean(g_pcaed(gender==0,:));
% female_im=mean(g_pcaed(gender==1,:));
% 
% for i=1:size(g,1)
% if(sum(g_pcaed(i,:)-male_im).^2 < sum(g_pcaed(i,:)-female_im).^2)
% soln(i,1)=0;
% else
% soln(i,1)=1;
% end;
% end


% 
% res=temp_im*c_i(1:60,:);
% temp_im=g*c_i(:,1:500);
% res=temp_im*c_i(1:500,:);
% imshow(uint8(reshape(res(1,:),[100 100])))
% imagesc(uint8(reshape(res(1,:),[100 100])))
% imshow(uint8(reshape(res(100,:),[100 100])))

[c_i s_i l_i]=pca(g);

count=1;
for i=1:size(yhatsvm)

    if(yhatsvm(i,1)==yhatsvm_wi(i,1) && yhatsvm_wi(i,1) == yhat_ada(i,1))
        idx_(count,1)=i;
        count=count+1;
    end;

end;

count=0;
for i=1:size(yhatsvm)

    if(yhatsvm(i,1)==yhatsvm_wi(i,1) && yhatsvm_wi(i,1) == yhat_ada(i,1) && yhatsvm_wi(i,1) == yhat_lda(i,1))
        count=count+1;
        submission(i,1)=yhatsvm(i,1);
        
    end;
    

end;

count


for i=1:size(words,1)
stemmed_w{i}= (porterStemmer(char(words(i))));
end;


y = (yhatsvm + yhatsvm_wi + yhat_ada + yhat_linearreg + yhat_lr + yhatldat_ + yhatsvm1t_ + yhatsvm2t_ + yhatlinrt_ + yhat_stw + yhat_lda)/11;
for i=1:size(y)
if(y(i,1)>=0.5)
    y(i,1)=1;
else
    y(i,1)=0;
end;
end;


%%

y = (yhat_lda_tt + yhat_svm2_tt+ yhat_svm1_tt)/3;



for i=1:size(y)

% if(y(i,1)==3 || y(i,1)==4)
%     if(yhat_svm1_tt(i,1)==yhat_svm2_tt(i,1) && yhat_svm1_tt(i,1)==yhat_ada_tt(i,1))
%         y(i,1) = yhat_svm1_tt(i,1)*7;
%     end;
% end;    
%     
% y(i,1) = (y(i,1)/7);
if(y(i,1)>=0.5)
    y(i,1)=1;
else
    y(i,1)=0;
end;
end;



%%
