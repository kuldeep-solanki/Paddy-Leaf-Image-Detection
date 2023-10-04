% clc
% close all
% clear all
% 
% % Load a binary mask (you should replace this with your own logic)
% binaryMask = imread('2 (1).jpg');
% 
% % Extract features from the binary mask (you should replace this with your own logic)
% meanValue = mean(binaryMask(:));
% stdValue = std(double(binaryMask(:)));
% featureVector = [meanValue, stdValue];
% 
% % Load a pre-trained SVM classifier (you should replace this with your own)
% load('Training_Data.mat');
% 
% % Classify using SVM
% result = predict(trainedSVM, featureVector);
% 
% % Display the classification result
% if result == 1
%     fprintf('Disease Detected\n');
% else
%     fprintf('Disease Not Detected\n');
% end

clc
close all
clear all

while (1 == 1)
    choice = menu('Paddy Leaf Disease Detection', 'Load Image', 'Close');

    if (choice == 1)
        % Load an input image
        [filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick an Image File');
        I = imread(fullfile(pathname, filename));

        % Process the image: Resize, Convert to HSV
        I = imresize(I, [1000, 260]); % Resize to your desired dimensions
        I = rgb2hsv(I); % Convert to HSV color space

        % Enhance contrast in the Value (V) channel
        I(:,:,3) = imadjust(I(:,:,3)); % Enhance contrast in the V channel

        % Display the processed image
        figure, imshow(hsv2rgb(I));
        title('Processed Image');

        %% Create Binary Mask
        % Implement your mask creation logic here
        % Example: Binary mask creation using a threshold
        binaryMask = I(:,:,3) > 0.5;

        % Display the binary mask
        figure, imshow(binaryMask);
        title('Binary Mask');

        %% Feature Extraction
        % Extract features from the binary mask or the original image
        % Implement your feature extraction logic here
        % Example: Calculate mean and standard deviation of the binary mask
        meanValue = mean(binaryMask(:));
        stdValue = std(double(binaryMask(:)));

        % Example feature vector
        featureVector = [meanValue, stdValue];
        
        % Here, you can add your SVM classification code if you have it
        % Replace 'result' with your classification result
        % Example: result = predict(trainedSVM, featureVector);

        % Display the classification result (if available)
        % if result == 1
        %     helpdlg('Disease Detected');
        %     disp('Disease Detected');
        % else
        %     helpdlg('Healthy Leaf');
        %     disp('Healthy Leaf');
        % end
        
        % For testing, you can print "Disease Detected" as a placeholder
        fprintf('Disease Detected\n');
    end

    if (choice == 2)
        close all;
        return;
    end
end
