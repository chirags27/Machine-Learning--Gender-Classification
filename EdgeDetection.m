function [edge_images] = EdgeDetection( train_data)
%EDGEDETECTION Summary of this function goes here
%   Detailed explanation goes here
    %images_gray_train = Gray_Images(train_data,100,100);
    size_train = size(train_data);
    num_rows = size_train(1);
    
    for i=1:num_rows
        %gray_img = 
        %resized_img = imresize(uint8(gray_img),[250,250]);
        edge_img = edge(reshape(train_data(i,:),[100 100]),'Canny');
        %resized_edge_img = imresize(edge_img,[100 100]);
        edge_images(i,1:100*100) = reshape(edge_img,[1 100*100]);
    end
end

