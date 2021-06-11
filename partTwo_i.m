clear; clc; close all;

% Obtaining the images where the bottle is underfilled
images_dir = 'Pictures\1-UnderFilled';
file_data = GetFileFromDirectory(images_dir);
num_images = length(file_data);

% Initialising the noise ranges and the number of times to execute the test
num_tests = 10;
noiseLevel = 0.5;
noise_ranges = 0.05:0.05:0.5;
output = zeros(length(noise_ranges));

% Iterating over the number of tests
for j = 1:num_tests
    % Iterating over the noise ranges
    for N = 1:length(noise_ranges)
        fault_count = 0;

        % Iterating over number of images
        for i = 1:num_images
            % Loading the image from the directory
            file_path = fullfile(images_dir , file_data(i).name);
            image = imread(file_path);

            % Adding Gaussian noise with a mean of '0' to the image
            noisy_image = imnoise(image, 'gaussian', 0, noiseLevel);
 
            % Applying average/mean filter to the noisy image
            %filtered_image = imfilter(noisy_image, ones(N, N)/N^2);
            
            % Applying median filter to the noisy image
            %filtered_image = medfilt2(rgb2gray(noisy_image), [N, N]);

            % Applying low-pass filter to the noisy image
            filtered_image = IdealLowPassFilt(noisy_image, noise_ranges(N));

             % Checking if the fault is detected
            bottleUnderfilled = Bottle_is_Underfilled(filtered_image);
               
            % Count the number of detected faults
            fault_count = fault_count + bottleUnderfilled;
        end

        % Add accuracy % results to the output array
        output(N) = output(N) + (100*(fault_count / num_images));
    end
end

% Dividing each element of the output array by the number of tests to get the average performance
output = output ./ num_tests;
%Generating kernelRange array for plotting alon the X-axis
kernelRange = 1:length(noise_ranges); 

% % Plot the Averaging Filter Performance
% figure;
% bar(kernelRange, output, 'r', 'BarWidth', 10);
% title('Performance - Mean Filter (\sigma = 0.5)');
% xlabel('Kernel Size (N)')
% ylabel('Accuracy %');
% ylim([0,  105])
% grid on;
% legend('Mean Filter', 'Location', 'northwest');

% % Plot the Median Filter Performance
% figure;
% bar(kernelRange, output, 'b', 'BarWidth', 10);
% title('Performance - Median Filter (\sigma = 0.5)');
% xlabel('Kernel Size ')
% ylabel('Accuracy %');
% ylim([0,  105])
% grid on;
% legend('Median Filter', 'Location', 'northwest');


% Plotting the LPF Performance
figure;
bar(noise_ranges, output, 'c', 'BarWidth', 10);
title('Fault Detection Performance (\sigma = 0.5)');
xlabel('Normalized Cutoff Frequency')
ylabel('Accuracy %');
ylim([0,  105])
grid on;
legend('Freq Domain Filter', 'Location', 'northeast');

function result = GetFileFromDirectory(Path)
% Check to make sure that folder actually exists.
if ~isfolder(Path)
    disp('no such directory exists');
    return;
end
% Get a list of all '.jpg' files in the directory
file_path = fullfile(Path, '*.jpg');
result = dir(file_path);
end


% Function to detect images in which the bottle is underfilled
function result = Bottle_is_Underfilled(image)
    % Converting the image to greyscale
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    % Extracting/cropping only the interested region of the image 
    interested_region = imcrop(image,[140,130,80,40]);
    % Convert to a binary image
    binary_image = imbinarize(interested_region, double(150/256));
    % Calculate the percentage of black pixels in the binary image
    black_pixels_percentage = sum(binary_image(:) == 0) / numel(binary_image(:));
    % The fault is recognised if percentage of black pixels < 0.25
    result = black_pixels_percentage < 0.25;
end


% function to apply a low-pass (frequency domain) filter to an image
% (Referenced from Lectures)
function result = IdealLowPassFilt(image, cutoffFreq)
    % Getting the image dimensions
    [l, b, channels] = size(image);
    % Getting the centered version of discrete Fourier transform
    A=fft2(image); 
    DFT_center=fftshift(A);
    % Calculating the centerpoint of the image
    hr = (l-1)/2; 
    hc = (b-1)/2; 
    [X, Y] = meshgrid(-hc:hc, -hr:hr);
    % Constructing an ideal low-pass filter
    freq_filt = sqrt((X/hc).^2 + (Y/hr).^2); 
    freq_filt = double(freq_filt <= cutoffFreq);
    % producing a  RGB output of the centered filter
    output_image = zeros(size(DFT_center)); 
    for channel = 1:channels 
        output_image(:, :, channel) = DFT_center(:, :, channel) .* freq_filt; 
    end 
    % Centered filter on the spectrum
    B = ifftshift(output_image);
    % Normalizing to the range [1, 256]
    result = uint8(256 * mat2gray(abs(ifft2(B))));
end