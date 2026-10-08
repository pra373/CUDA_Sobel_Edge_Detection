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

void uninitialize(void);

int main(void)
{
	logFile = fopen("../log/log.txt", "w");

	if (logFile == nullptr)
	{
		cout << "Failed to open log file !" << endl;
		getchar();
		exit(EXIT_FAILURE);
	}

	fprintf(logFile, "Program started successfully !\n");

	imageData = stbi_load("../resources/lotus.jpg", &width, &height, &channels, 3);

	if (imageData != nullptr)
	{
		fprintf(logFile, "Image Data Loaded Successfully !\n");
		fprintf(logFile, "Width:\t%d\nHeight:\t%d\nChannels:%d\n",width, height, channels);
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

	// launch kernel here

	launchGreyScaleKernel(dev_imageData, dev_greyScaleOutputImage, width, height);

	// copy grey scale output image to cpu side

	HANDLE_ERROR(cudaMemcpy(outputGreyScaleImage, dev_greyScaleOutputImage, totalSizeOfImageInBytes, cudaMemcpyDeviceToHost));

	//write grey scale output image 

	stbi_write_jpg("../output/Grey_Scale_Lotus.jpg", width, height, 3, outputGreyScaleImage, 100);

	// allocate GPU and CPU buffer for sobel output image

	outputSobelImage = (unsigned char*)malloc(totalSizeOfImageInBytes);

	if (outputSobelImage == nullptr)
	{
		fprintf(logFile, "host side memory allocation for sobel output image failed !");
		exit(EXIT_FAILURE);
	}

	HANDLE_ERROR(cudaMalloc((void**)&dev_sobelOutputImage, totalSizeOfImageInBytes));

	launchSobelKernel(dev_greyScaleOutputImage, dev_sobelOutputImage, width, height);

	//HANDLE_ERROR(cudaMemcpy(outputSobelImage, dev_sobelOutputImage, totalSizeOfImageInBytes, cudaMemcpyDeviceToHost))


	uninitialize();


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