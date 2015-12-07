function predictions = make_final_prediction(model,X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model
%predictions = X_test(:,1:5000) * model.w_ridge;
%predictions(predictions > 0.5) = 1;
%predictions(predictions <= 0.5) = 0;


words_test = X_test(:,1:5000);
img_feat_test = X_test(:,5001:end);

num_words_features = 600;
num_words_lda_logistic = 500;

words_normalized_test = bsxfun(@rdivide,words_test,sum(words_test.^2,2).^0.5);

word_test_reduced = words_normalized_test(:,index);

img_feat_test_reduced = img_feat_test(:,[1,2]);

mean_test_words = bsxfun(@minus,word_test_reduced,mean(word_test_reduced));

mean_image_features = bsxfun(@minus,img_feat_test_reduced,mean(img_feat_test_reduced));
%% Train models

X_test_svm1 = [mean_test_words(:,1:num_words_features) mean_image_features];
X_test_svm2 = word_test_reduced(:,1:num_words_features);
X_test_ada = [mean_test_words(:,1:num_words_features) mean_image_features];
X_test_lda = [mean_test_words(:,1:num_words_lda_logistic) mean_image_features];

%% Test models

X_test_svm1_test = [mean_test_words(:,1:num_words_features) mean_image_features];
X_test_svm2_test = word_test_reduced(:,1:num_words_features);
X_test_lda_test = [mean_test_words(:,1:num_words_features) mean_image_features];
X_test_linear_test = [mean_test_words(:,1:num_words_lda_logistic) mean_image_features];

%%

yhatsvm = model.svm_model1_train.predict(X_test_svm1);
yhatsvm_wi = model.svm_model2_train.predict(X_test_svm2);
yhat_ada = model.ada_model_train.predict(X_test_ada);
yhat_lda = model.lda_model_train.predict(X_test_lda);

yhatsvm1t_=model.svm_model1_test.predict(X_test_svm1_test);
yhatsvm2t_=model.svm_model2_test.predict(X_test_svm2_test);
yhatldat_=model.lda_model_test.predict(X_test_lda_test);

yhatlinrt_=X_test_linear_test*model.lin_model_test;
yhatlinrt_(yhatlinrt_ >= 0.5) = 1;
yhatlinrt_(yhatlinrt_ < 0.5) = 0;



% n = numel(X_test(:,1));
% num_models = 7;
% predictions = zeros(n,1);
% 
% for i=1:n
%     predictions(i) = svm1_predictions(i)+svm2_predictions(i)...
%                      + stepwise_predictions(i) + lm_predictions(i)...
%                      + ada_predictions(i)+ lda_predictions(i)...
%                      + logistic_predictions(i);
%                  
% end
% 
% predictions(predictions >= num_models/2) = 1;
% predictions(predictions < num_models/2) = 0;



submission = zeros(X_test,1);
c1=0;
c2=0;
c3=0;
c4=0;
c5=0;
for i=1:size(submission,1)
    
    if(yhatsvm(i,1)== yhatsvm_wi(i,1) && (yhatsvm(i,1)== yhatsvm1t_(i,1)) && yhatsvm(i,1)== yhatsvm2t_(i,1)&& yhatsvm(i,1)== yhatldat_(i,1) && yhatsvm(i,1)== yhatlinrt_(i,1) && yhatsvm(i,1)== yhat_ada(i,1) && yhatsvm(i,1)== yhat_lda(i,1))
        submission(i,1)=yhatsvm(i,1);
        c1=c1+1;
    elseif(yhatsvm(i,1)== yhatsvm_wi(i,1) && (yhatsvm(i,1)== yhatsvm1t_(i,1)) && yhatsvm(i,1)== yhatsvm2t_(i,1))
        submission(i,1) = yhatsvm(i,1);
        c2=c2+1;
    elseif(yhatsvm(i,1)== yhatsvm_wi(i,1) && yhatsvm(i,1)== yhat_ada(i,1) && yhatsvm(i,1)== yhat_lda(i,1))
        submission(i,1) = yhatsvm(i,1);
          c3=c3+1;
     elseif(yhatsvm1t_(i,1)== yhatsvm2t_(i,1) && yhatsvm1t_(i,1)== yhatlinrt_(i,1) && yhatsvm1t_(i,1)== yhatldat_(i,1))
        submission(i,1) = yhatsvm1t_(i,1);
        c4=c4+1;
     else
        submission(i,1) = yhatsvm2t_(i,1);
        c5=c5+1;
    end;
end;

predictions = submission;

