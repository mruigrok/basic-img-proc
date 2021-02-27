#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <cmath>

/*
	Pixel conventions in RGB and YUV space
	R -> red, G -> green and B -> blue when in RGB space
	Y -> red, U -> green and V -> blue when in YUV space
*/

// Can be nagative values when converting to YUV space so use int type for this and unsigned char for RGB space
typedef union{
	unsigned char rgb;
	int yuv;
}pixel_type;

// This struct is to hold the RGB or YUV values for each pixel
typedef struct{
	pixel_type red;
	pixel_type green;
	pixel_type blue;
}pixel;

// 2D vector image structure
typedef std::vector<std::vector<pixel>> Image;

// Clamp the pixel values b/w upper and lower bound (typcially 0-255)
inline int clamp(int value, int lower, int upper)
{
	if (value < lower) return lower;
	else if (value > upper) return upper;
	else return value;
}
 
Image read_in_image(const std::string filename)
{
	cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
	int n = image.rows;
	int m = image.cols;

	/* Initialize vector for the image, add in the values from read in image */
	Image image_matrix(n, std::vector<pixel>(m));
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{	
			image_matrix[i][j].red.rgb = image.at<cv::Vec3b>(i, j)[2];
			image_matrix[i][j].green.rgb = image.at<cv::Vec3b>(i, j)[1];
			image_matrix[i][j].blue.rgb = image.at<cv::Vec3b>(i, j)[0];
		}
	}
	return image_matrix;
}

// Change each pixel to (Y, U, V) with convention red as Y, green as U and blue as V
void convert_rgb_to_yuv(Image& image)
{
	std::vector<std::vector<pixel>>::iterator row;
	std::vector<pixel>::iterator col;
	for (row = image.begin(); row != image.end(); ++row)
	{
		for (col = row->begin(); col != row->end(); ++col)
		{
			unsigned char r = col->red.rgb;
			unsigned char g = col->green.rgb;
			unsigned char b =col->blue.rgb;

			col->red.yuv = (int)round(0.299 * r + 0.587 * g + 0.114 * b);
			col->green.yuv = (int)round(-0.14713 * r - 0.28886 * g + 0.436 * b);
			col->blue.yuv = (int)round(0.615 * r - 0.51499 * g - 0.10001 * b);
		}
	}
	std::cout << "Done converting to YUV!" << std::endl;
	return;
}

// Change each pixel to (R, G, B) with convention red as R, green as G and blue as B //
void convert_yuv_to_rgb(Image& image)
{
	std::vector< std::vector<pixel> >::iterator row;
	std::vector<pixel>::iterator col;
	for (row = image.begin(); row != image.end(); ++row)
	{
		for (col = row->begin(); col != row->end(); ++col)
		{
			int y = col->red.yuv;
			int u = col->green.yuv;
			int v = col->blue.yuv;

			col->red.rgb = (unsigned char)clamp((int)round(y + 1.13983 * v), 0, 255);
			col->green.rgb = (unsigned char)clamp((int)round(y - 0.39465 * u - 0.58060 * v), 0, 255);
			col->blue.rgb = (unsigned char)clamp((int)round(y + 2.03211 * u), 0, 255);
		}
	}
	std::cout << "Done converting to RGB!" << std::endl;
	return;
}

// s = c * r ^ gamma transformation function - spatial domain
void gamma_correction(Image& image, double gamma, double c)
{
	std::vector<std::vector<pixel>>::iterator row;
	std::vector<pixel>::iterator col;
	for (row = image.begin(); row != image.end(); ++row)
	{
		for (col = row->begin(); col != row->end(); ++col)
		{
			int s = (int)round(c * pow((double)clamp(col->red.yuv, 0, 255), gamma));
			col->red.yuv = clamp(s, 0, 255);
		}
	}
	return;
}

// s = c log(1 + r) transfer function
void log_transformation(Image& image, double c)
{
	std::vector<std::vector<pixel>>::iterator row;
	std::vector<pixel>::iterator col;
	for (row = image.begin(); row != image.end(); ++row)
	{
		for (col = row->begin(); col != row->end(); ++col)
		{
			int s = (int)round(c * log10(clamp(1 + col->red.yuv, 0, 255)));
			col->red.yuv = clamp(s, 0, 255);
		}
	}
	return;
}

void piecewise_linear(Image& image, double alpha, double beta, double gamma, int r1, int r2)
{
	if (r1 > r2) return;
	std::vector<std::vector<pixel>>::iterator row;
	std::vector<pixel>::iterator col;
	for (row = image.begin(); row != image.end(); ++row)
	{
		for (col = row->begin(); col != row->end(); ++col)
		{
			int s;
			int r = clamp(col->red.yuv, 0, 255);
			int s1 = (int)round(alpha * r1);
			int s2 = (int)round(beta * (r2 - (int)r1) + s1);
			if (r <= r1)
				s = (int)round(alpha * r);
			else if (r <= r2)
				s = (int)round(beta * (r - (int)r1) + s1);
			else
				s = (int)round(gamma * (r - (int)r2) + s2);
			col->red.yuv = clamp(s, 0, 255);
		}
	}
	return;
}

// Histogram equalization image correction - spatial domain
void histogram_equilization(Image& image)
{
	// Create a vector to hold the pixel intensity counts (only the Y pixel)
	int L = 256; // 0-255 pixel values 
	std::vector<int> intensity_count(L, 0);

	// Find histogram of pixels, clamped between 0-255
	std::vector<std::vector<pixel>>::iterator row;
	std::vector<pixel>::iterator col;
	for (row = image.begin(); row != image.end(); ++row) 
		for (col = row->begin(); col != row->end(); ++col) 
			intensity_count[clamp(col->red.yuv, 0, 255)] += 1;
	
	// Keep a sum of the pixel counts for computing the mapping function
	int sum = 0;
	int NM = image.size() * image[0].size();
	double sum_const = (double)(L - 1) / NM;

	//Compute the mapped pixel value and add to the mapped values
	std::vector<int> pixel_map(L, 0);
	std::vector<int>::iterator it;
	for (it = intensity_count.begin(); it != intensity_count.end(); ++it)
	{
		sum += *it;
		double pixel_val = sum * sum_const;
		pixel_map[it - intensity_count.begin()] = clamp((int)round(pixel_val), 0, 255);
	}

 	// Go through entire image and get the new pixel value from the pixel map (r -> s)
	for (row = image.begin(); row != image.end(); ++row)
	{
		for (col = row->begin(); col != row->end(); ++col)
		{
			col->red.yuv = pixel_map[clamp(col->red.yuv, 0, 255)];	
		}
	}
	return;
}

// Get the Fourier of the image given by filename
cv::Mat read_in_image_fourier(const std::string filename)
{
	cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		std::cout << "Error opening image" << std::endl;
		return image;
	}

	cv::Mat padded;
	int n = cv::getOptimalDFTSize(image.rows);
	int m = cv::getOptimalDFTSize(image.cols);

	// Set up the padding and 2 images (real and imaginary values)
	cv::copyMakeBorder(image, padded, 0, n - image.rows, m - image.cols, cv::BORDER_CONSTANT, 0);
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complex_image;
	cv::merge(planes, 2, complex_image);

	// Compute the DFT -  planes[0] = real, planes[1] = complex
	dft(complex_image, complex_image);
	cv::split(complex_image, planes);
	cv::magnitude(planes[0], planes[1], planes[0]);
	cv::Mat mag_image = planes[0];

	// Switch to the log scale
	mag_image += cv::Scalar::all(1);
	log(mag_image, mag_image);

	// Crop the spectrum
	mag_image = mag_image(cv::Rect(0, 0, mag_image.cols & -2, mag_image.rows & -2));

	int cx = mag_image.cols / 2;
	int cy = mag_image.rows / 2;
	cv::Mat q0(mag_image, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(mag_image, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(mag_image, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(mag_image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	// Swap quadrants, Top-Left with Bottom-Right and Top-Right with Bottom-Left
	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                   
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// Normalize and return
	cv::normalize(mag_image, mag_image, 0, 1, cv::NormTypes::NORM_MINMAX);
	
	// Copy into image data structur or keep in Mat format *****

	cv::namedWindow("spectrum magnitude", cv::WindowFlags::WINDOW_NORMAL);
	cv::resizeWindow("spectrum magnitude", 2048, 1000);
	cv::imshow("spectrum magnitude", mag_image);

	cv::imshow("Input Image", image);    // Show the result
	cv::waitKey();

	return mag_image;
}

void save_image(const std::string filename, Image& image)
{
	cv::Mat processed_image(image.size(), image[0].size(), CV_8UC3);
	for (int i = 0; i < image.size(); ++i)
	{
		for (int j = 0; j < image[0].size(); ++j)
		{
			// Add pixel values to a cv:mat
			processed_image.at<cv::Vec3b>(i, j)[2] = (unsigned char)image[i][j].red.rgb;
			processed_image.at<cv::Vec3b>(i, j)[1] = (unsigned char)image[i][j].green.rgb;
			processed_image.at<cv::Vec3b>(i, j)[0] = (unsigned char)image[i][j].blue.rgb;
		}
	}
	// Display image
	cv::namedWindow("image", cv::WindowFlags::WINDOW_NORMAL);
	cv::imshow("image", processed_image);
	cv::waitKey(0);
	// Write to file
	cv::imwrite("output_2.jpg", processed_image);
}

// Use to compare to defined
void open_cv_he()
{
	// Read the image file
	cv::Mat image = cv::imread("C:\\Users\\Matt Ruigrok\\Documents\\School\\Year 4\\CE 4TN4\\Project 1\\10087_00_30s.jpg");

	// Check for failure
	if (image.empty())
	{
		std::cout << "Could not open or find the image" << std::endl;
		std::cin.get(); //wait for any key press
		return;
	}

	//Convert the image from BGR to YCrCb color space
	cv::Mat hist_equalized_image;
	cv::cvtColor(image, hist_equalized_image, cv::ColorConversionCodes::COLOR_BGR2YCrCb);

	//Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
	std::vector<cv::Mat> vec_channels;
	cv::split(hist_equalized_image, vec_channels);

	//Equalize the histogram of only the Y channel 
	cv::equalizeHist(vec_channels[0], vec_channels[0]);

	//Merge 3 channels in the vector to form the color image in YCrCB color space.
	cv::merge(vec_channels, hist_equalized_image);

	//Convert the histogram equalized image from YCrCb to BGR color space again
	cvtColor(hist_equalized_image, hist_equalized_image, cv::ColorConversionCodes::COLOR_YCrCb2BGR);

	cv::imwrite("opencv_output.jpg", hist_equalized_image);
	return;
}

int main()
{
	// Get the specified image filename, read in an then correct the image
	const std::string filename = "C:\\Users\\Matt Ruigrok\\Documents\\School\\Year 4\\CE 4TN4\\Project 1\\10087_00_30s.jpg";
	//const std::string filename = "C:\\Users\\Matt Ruigrok\\Documents\\School\\Year 4\\CE 4TN4\\diag_bricks.jpg";

	Image image_matrix = read_in_image(filename);
	//cv::Mat image = read_in_image_fourier(filename);
	std::cout << +image_matrix[0][0].red.rgb << std::endl;
	std::cout << +image_matrix[0][0].green.rgb << std::endl;
	std::cout << +image_matrix[0][0].blue.rgb << std::endl;

	// Convert from RGB to YUV space
	convert_rgb_to_yuv(image_matrix);
	std::cout << image_matrix[0][0].red.yuv << std::endl;
	std::cout << image_matrix[0][0].green.yuv << std::endl;
	std::cout << image_matrix[0][0].blue.yuv << std::endl;

	// Now apply the filter !!
	//gamma_correction(image_matrix, 0.6, 8);
	//log_transformation(image_matrix, 100);
	//piecewise_linear(image_matrix, 5, 0.6, 0.25, 20, 200);
	histogram_equilization(image_matrix);
	std::cout << image_matrix[0][0].red.yuv << std::endl;
	std::cout << image_matrix[0][0].green.yuv << std::endl;
	std::cout << image_matrix[0][0].blue.yuv << std::endl;

	// Now convert back to RGB from YUV space
	convert_yuv_to_rgb(image_matrix);
	std::cout << +image_matrix[0][0].red.rgb << std::endl;
	std::cout << +image_matrix[0][0].green.rgb << std::endl;
	std::cout << +image_matrix[0][0].blue.rgb << std::endl;

	// Save processed image to file
	save_image(filename, image_matrix);
}


