clear; clc; close all;

% Obtaining the images where the bottle cap is missing
BCM_imagesDir = 'Pictures\6-CapMissing';
BCM_fileData = GetFileFromDirectory(BCM_imagesDir);

% Obtaining the images where the bottle is underfilled
BU_imagesDir = 'Pictures\1-UnderFilled';
BU_fileData = GetFileFromDirectory(BU_imagesDir);

% total number of images
num_images = length(BCM_fileData);

% Initialising the noise ranges and the number of times to execute the test
noise_ranges = 0.0:0.005:0.35;
num_tests = 5;
output = zeros(2, length(noise_ranges));

% Looping over the number of tests
for k = 1:num_tests
    % Looping over the noise Ranges
    for j = 1:length(noise_ranges)
        % Initializing fault counts
        BCM_faultCount = 0;
        BU_faultCount = 0;
    
        % Looping over the number of Images
        for i = 1:num_images
            
            % BOTTLE_CAP_MISSING
            % Loading the image from the directory
            filePath = fullfile(BCM_imagesDir, BCM_fileData(i).name);
            image = imread(filePath);
            % Adding Gaussian noise with a mean of '0' to the image
            imageWithNoise = imnoise(image, 'gaussian', 0, noise_ranges(j));
            
            % Checking if the fault is detected
            bottleCapMissing = BottleCap_is_Missing(imageWithNoise);
            % Updating the fault count accordingly
            BCM_faultCount = BCM_faultCount + bottleCapMissing;
            

            % BOTTLE_UNDERFILLED
            % Loading the image from the directory
            filePath = fullfile(BU_imagesDir, BU_fileData(i).name);
            image = imread(filePath);
            % Adding Gaussian noise with a mean of '0' to the image
            imageWithNoise = imnoise(image, 'gaussian', 0, noise_ranges(j));
            
            % Checking if the fault is detected
            bottleUnderfilled = Bottle_is_Underfilled(imageWithNoise);
            % Updating the fault count accordingly
            BU_faultCount = BU_faultCount  + bottleUnderfilled;
          
        end
        
        % Add accuracy % results to the output array
        output(1, j) = output(1, j) + (100*(BCM_faultCount/num_images));
        output(2, j) = output(2, j) + (100*(BU_faultCount/num_images));
    end
end

% Dividing each element of the output array by the number of tests to get the average performance
output = output ./ num_tests;

% ----------------------------------------------------------------
% BOTTLE_CAP_MISSING
% Plot bar chart
figure;
bar(noise_ranges, output(1, :), 1, 'r');
title('Fault Detection Performance');
xlabel('Noise Range')
ylabel('Accuracy %');
ylim([0,  105])
grid on;
legend('Bottle Cap Missing');


% BOTTLE_UNDERFILLED
% Plot bar chart
figure;
bar(noise_ranges, output(2, :), 1, 'b');
title('Fault Detection Performance');
xlabel('Noise Range')
ylabel('Accuracy %');
ylim([0,  105])
grid on;
legend('Bottle Underfilled');


% OVERALL_PERFORMANCE
% Plot graph
figure;
plot(noise_ranges, output(1, :), 'r', 'LineWidth', 2); hold on;
plot(noise_ranges, output(2, :), 'b', 'LineWidth', 2); hold on;
title('Fault Detection Performance');
xlabel('Noise Range')
ylabel('Accuracy %');
ylim([0,  105])
grid on;
legend('Bottle Cap Missing', 'Bottle Underfilled');


function result = GetFileFromDirectory(Path)
% Check to make sure that folder actually exists.
if ~isfolder(Path)
    disp('no such directory exists');
    return;
end
% Get a list of all '.jpg' files in the directory
filePath = fullfile(Path, '*.jpg');
result = dir(filePath);
end



% Function to detect images in which the bottle cap is missing
function result = BottleCap_is_Missing(image)
    % Converting the image to greyscale
    image = rgb2gray(image);
    % Extracting/cropping only the interested region of the image 
    interested_region = imcrop(image,[150,5,50,40]);
    % Convert to a binary image 
    binary_image = imbinarize(interested_region, double(150/256));
    % Calculate the percentage of black pixels in the binary image
    black_pixels_percentage = (sum(binary_image(:) == 0) / numel(binary_image(:))); 
    % The fault is recognised if percentage of black pixels < 0.25
    result = black_pixels_percentage < 0.25;
end



% Function to detect images in which the bottle is underfilled
function result = Bottle_is_Underfilled(image)
    % Converting the image to greyscale
    image = rgb2gray(image);
    % Extracting/cropping only the interested region of the image 
    interested_region = imcrop(image,[140,130,80,40]);
    % Convert to a binary image
    binary_image = imbinarize(interested_region, double(150/256));
    % Calculate the percentage of black pixels in the binary image
    black_pixels_percentage = sum(binary_image(:) == 0) / numel(binary_image(:));
    % The fault is recognised if percentage of black pixels < 0.25
    result = black_pixels_percentage < 0.25;
end
