clear; clc; close all;

% Obtaining the images where the bottle is underfilled
images_dir = 'Pictures\1-UnderFilled';
file_data = GetFileFromDirectory(images_dir);
num_images = length(file_data);

% Initialising the noise ranges and the number of times to execute the test
num_tests = 10;
noise_ranges = 0.0:0.01:1.0;
output = zeros(4, length(noise_ranges));

% Iterating over the number of tests
for k = 1:num_tests
    % Loop over the noise levels
    for j = 1:length(noise_ranges)
        faultCount = 0;
        Mean_faultCount = 0;
        Median_faultCount = 0;
        LPF_faultCount = 0;
        
        % Iterating over the number of images
        for i = 1:num_images
            % Loading the image from the directory
            filePath = fullfile(images_dir, file_data(i).name);
            image = imread(filePath);

            % Adding Gaussian noise with a mean of '0' to the image
            noisy_image = imnoise(image, 'gaussian', 0, noise_ranges(j));

            % Applying average/mean filter to the noisy image
            Mean_filtered_image = imfilter(noisy_image, ones(N, N)/N^2);

            % Applying median filter to the noisy image
            Median_filtered_image = medfilt2(rgb2gray(noisy_image), [N, N]);

            % Applying low-pass filter to the noisy image
            Freq_filtered_image = IdealLowPassFilt(noisy_image, 0.1);

            % Check if fault detected
            bottle_underfilled      = Bottle_is_Underfilled(noisy_image);
            Mean_bottle_underfilled   = Bottle_is_Underfilled(Mean_filtered_image);
            Median_bottle_underfilled    = Bottle_is_Underfilled(Median_filtered_image);
            Freq_bottle_underfilled   = Bottle_is_Underfilled(Freq_filtered_image);

            faultCount      = faultCount      + bottle_underfilled;
            Mean_faultCount  = Mean_faultCount  + Mean_bottle_underfilled;
            Median_faultCount   = Median_faultCount   + Median_bottle_underfilled;
            LPF_faultCount  = LPF_faultCount  + Freq_bottle_underfilled ;
        end
        
        % Add fault classifcation % to the output array
        output(1, j) = output(1, j) + (100*(faultCount  / num_images));
        output(2, j) = output(2, j) + (100*(Mean_faultCount  / num_images));
        output(3, j) = output(3, j) + (100*(Median_faultCount   / num_images));
        output(4, j) = output(4, j) + (100*(LPF_faultCount  / num_images));
    end
end

% Dividing each element of the output array by the number of tests to get the average performance
output = output ./ num_tests;


% Plot graph
figure;
plot(noise_ranges, output(1, :), 'k', 'LineWidth', 2); hold on;
plot(noise_ranges, output(2, :), 'b', 'LineWidth', 2); hold on;
plot(noise_ranges, output(3, :), 'm', 'LineWidth', 2); hold on;
plot(noise_ranges, output(4, :), 'y', 'LineWidth', 2); hold on;
title('Fault Detection Performance');
xlabel('Noise Level')
ylabel('Accuracy %');
ylim([0,  105])
grid on;
legend({'No Filter', 'Mean', 'Median', 'Low-Pass'}, 'Location', 'south');


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