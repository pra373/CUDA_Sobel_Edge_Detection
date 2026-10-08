#pragma once

__global__ void greyScaleKernel(unsigned char* dev_imageData, unsigned char* dev_greyScaleOutputImage, int width, int height);
__global__ void sobelKernel(unsigned char* dev_imageData, unsigned char* dev_sobelOutputImage, int width, int height);
void launchGreyScaleKernel(unsigned char* dev_imageData, unsigned char* dev_greyScaleOutputImage, int width, int height);
void launchSobelKernel(unsigned char* dev_imageData, unsigned char* dev_sobelOutputImage, int width, int height);