#include<iostream>
#include<cuda_runtime.h>
#include"../include/kernels.h"

#define STB_IMAGE_IMPLEMENTATION
#include"../Include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Include/stb_image_write.h"


#define HANDLE_ERROR(call)                                                    \
do                                                                          \
{                                                                           \
    cudaError_t error = (call);                                             \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        fprintf(logFile, "CUDA Error: %s\n", cudaGetErrorString(error));     \
        fprintf(logFile, "File: %s\n", __FILE__);                            \
        fprintf(logFile, "Line: %d\n", __LINE__);                            \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

using std::cout;
using std::endl;

FILE* logFile = nullptr;

unsigned char* imageData = nullptr;
unsigned char* outputGreyScaleImage = nullptr;
int width, height, channels;

unsigned char* dev_imageData = nullptr;
unsigned char* dev_greyScaleOutputImage = nullptr;

unsigned char* outputSobelImage = nullptr;
unsigned char* dev_sobelOutputImage = nullptr;

//sobel arrays

int host_sobelGx[9] = { -1, 0 ,1, -2, 0 ,2, -1, 0 , 1 };
int host_sobelGy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

void uninitialize(void);

int main(void)
{
	cudaEvent_t start, stop;

	logFile = fopen("../log/log.txt", "w");

	if (logFile == nullptr)
	{
		cout << "Failed to open log file !" << endl;
		getchar();
		exit(EXIT_FAILURE);
	}

	fprintf(logFile, "Program started successfully !\n");
	cout << "Program started successfully !" << endl << endl;

	imageData = stbi_load("../resources/rose.jpg", &width, &height, &channels, 3);

	if (imageData != nullptr)
	{
		fprintf(logFile, "Image Data Loaded Successfully !\n");
		cout << "Image data loaded successfully !" << endl;
		fprintf(logFile, "Width:\t%d\nHeight:\t%d\nChannels:%d\n",width, height, channels);
		cout << "Width:\t" << width << "\nHeight:\t" << height << "\nChannels:" << channels << endl << endl;
	}

	//Allocate GPU buffer for input image and output image (grey scale)

	size_t totalSizeOfImageInBytes = width * height * 3 * sizeof(char);

	HANDLE_ERROR(cudaMalloc((void**)&dev_imageData, totalSizeOfImageInBytes));
	HANDLE_ERROR(cudaMalloc((void**)&dev_greyScaleOutputImage, totalSizeOfImageInBytes));

	// allocate host memory for grey scale image output

	outputGreyScaleImage = (unsigned char*)malloc(totalSizeOfImageInBytes);

	if (outputGreyScaleImage == nullptr)
	{
		fprintf(logFile, "Allocation of host memory for output greyscale image failed !!");
		exit(EXIT_FAILURE);
	}

	//send image data from Host to device

	HANDLE_ERROR(cudaMemcpy(dev_imageData, imageData, totalSizeOfImageInBytes, cudaMemcpyHostToDevice));

	//use cuda events to calculate execution time of kernel

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start));

	// launch greyScale kernel here

	launchGreyScaleKernel(dev_imageData, dev_greyScaleOutputImage, width, height);

	HANDLE_ERROR(cudaGetLastError());

	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	fprintf(logFile, "Grayscale kernel executed successfully!\n");
	cout << "GreyScale kernel executed successfully ! " << endl;

	float grayscaleTime = 0.0f;

	HANDLE_ERROR(cudaEventElapsedTime(&grayscaleTime, start, stop));

	fprintf(logFile, "Grayscale kernel execution time: %.4f ms\n", grayscaleTime);
	cout << "Grayscale kernel execution time: " << grayscaleTime << " ms" << endl;

	// copy grey scale output image to cpu side

	HANDLE_ERROR(cudaMemcpy(outputGreyScaleImage, dev_greyScaleOutputImage, totalSizeOfImageInBytes, cudaMemcpyDeviceToHost));

	//write grey scale output image 

	int result_greyScale = stbi_write_jpg("../output/Grey_Scale_Rose.jpg", width, height, 3, outputGreyScaleImage, 100);

	fprintf(logFile, "Grey scale image write result: %d\n", result_greyScale);
	cout << "Grey scale image write result: " << result_greyScale << endl << endl;


	// allocate GPU and CPU buffer for sobel output image

	outputSobelImage = (unsigned char*)malloc(totalSizeOfImageInBytes);

	if (outputSobelImage == nullptr)
	{
		fprintf(logFile, "host side memory allocation for sobel output image failed !");
		exit(EXIT_FAILURE);
	}

	HANDLE_ERROR(cudaMalloc((void**)&dev_sobelOutputImage, totalSizeOfImageInBytes));

	//copy constant memory arrays from host to device

	copySobelConstantsToDevice(host_sobelGx, host_sobelGy);

	//use cuda events to calculate execution time of kernel

	HANDLE_ERROR(cudaEventRecord(start));

	// launch sobel kernel here

	launchSobelKernel(dev_greyScaleOutputImage, dev_sobelOutputImage, width, height);

	HANDLE_ERROR(cudaGetLastError());

	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	fprintf(logFile, "Sobel kernel executed successfully!\n");
	cout << "Sobel kernel executed successfully ! " << endl;

	float sobelTime = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&sobelTime, start, stop));

	fprintf(logFile, "Sobel kernel execution time: %.4f ms\n", sobelTime);
	cout << "Sobel kernel execution time: " << sobelTime << " ms" << endl;
	
	// copy output sobel image from device to host

	HANDLE_ERROR(cudaMemcpy(outputSobelImage, dev_sobelOutputImage, totalSizeOfImageInBytes, cudaMemcpyDeviceToHost));

	// write sobel output image
	
	int result_sobel = stbi_write_jpg("../output/Sobel_rose.jpg", width, height, 3, outputSobelImage, 100);

	fprintf(logFile, "Sobel image write result: %d\n", result_sobel);
	cout << "Sobel image write result: " << result_greyScale << endl;

	uninitialize();

	getchar();

	return(0);
}

void uninitialize(void)
{

	if (dev_sobelOutputImage)
	{
		cudaFree(dev_sobelOutputImage);
		dev_sobelOutputImage = nullptr;
	}

	if (outputSobelImage)
	{
		free(outputSobelImage);
		outputSobelImage = nullptr;
	}

	if (outputGreyScaleImage)
	{
		free(outputGreyScaleImage);
		outputGreyScaleImage = nullptr;
	}

	if (dev_greyScaleOutputImage)
	{
		cudaFree(dev_greyScaleOutputImage);
		dev_greyScaleOutputImage = nullptr;
	}

	if (dev_imageData)
	{
		cudaFree(dev_imageData);
		dev_imageData = nullptr;
	}

	if (imageData)
	{
		stbi_image_free(imageData);
		imageData = nullptr;
	}

	if (logFile)
	{
		fclose(logFile);
		logFile = nullptr;
	}
}