for i=1:size(g,1)
    eq_im(i,:)=uint8(reshape(histeq(uint8(reshape(g(i,:),[100 100]))),[1 10000]));
end;

for i=1:size(eq_im,1)
hist_images(i,:) = (imhist(eq_im(i,:)))';
end;
temp_im_h = mean(hist_images);
hist_final=(hist_images(:,temp_im_h~=0));
hist_final = bsxfun(@rdivide,hist_final,sum(histfinal.^2,2).^0.5);
hist_final= meanCentreFeatures(hist_final);
%Comsidering 1500 important words
[accsvm_cv_im_h]=cross_validation(@svmtrainmatlab,[impwords_w meanCentreFeatures(images_transformed(:,[1 2 3 4]),Ytrain,4000);



%edge_images=EdgeDetection(eq_im);
%edge_images=double(edge_images)*255;
edge_images=double(canny_edge_eq);
male_im=(mean(edge_images(gender==0,:)));
female_im=(mean(edge_images(gender==1,:)));
colormap gray;
imagesc((reshape(male_im,[100 100])));
imagesc((reshape(female_im,[100 100])));

for i=1:size(eq_im,1)
if(corr2(eq_im(i,:),male_im) > corr2(eq_im(i,:),female_im))
soln(i,1)=0;
else
soln(i,1)=1;
end;
end
for i=1:size(eq_im,1)
canny_edge_eq(i,:) = edge(reshape(eq(i,:),[100 100]),'canny');
end;
[c_i s_i l_i]=pca(eq_im);



