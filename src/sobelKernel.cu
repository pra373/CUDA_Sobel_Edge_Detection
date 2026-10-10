#include<math.h>

__constant__ int sobelGx[9];
__constant__ int sobelGy[9];
__global__ void sobelKernel(unsigned char* dev_imageData, unsigned char* dev_sobelOutputImage, int width, int height)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx >= width || idy >= height)
	{
		return;
	}

	int offset = idx + idy * width;
	int RGBOffset = offset * 3;

	int Gx = 0;
	int Gy = 0;


	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int neighborX = idx + (j - 1);
			int neighborY = idy + (i - 1);

			neighborX = max(0, min(neighborX, width - 1));
			neighborY = max(0, min(neighborY, height - 1));

			int neighbourOffset = neighborX + neighborY * width;

			int neighbourRGBOffset = neighbourOffset * 3;

			int pixelValue = dev_imageData[neighbourRGBOffset];

			Gx += pixelValue * sobelGx[i * 3 + j];
			Gy += pixelValue * sobelGy[i * 3 + j];
		}
	}

	float magnetude = sqrtf((Gx * Gx) + (Gy * Gy));

	if (magnetude > 60)
	{
		dev_sobelOutputImage[RGBOffset] = 255;
		dev_sobelOutputImage[RGBOffset + 1] = 255;
		dev_sobelOutputImage[RGBOffset + 2] = 255;
	}
	else
	{
		dev_sobelOutputImage[RGBOffset] = 0;
		dev_sobelOutputImage[RGBOffset + 1] = 0;
		dev_sobelOutputImage[RGBOffset + 2] = 0;
	}
}

void launchSobelKernel(unsigned char* dev_imageData, unsigned char* dev_sobelOutputImage, int width, int height)
{
	dim3 blockDim(32, 32);
	dim3 gridDim((width + (blockDim.x - 1)) / blockDim.x, (height + (blockDim.y - 1)) / blockDim.y);

	sobelKernel << <gridDim, blockDim >> > (dev_imageData, dev_sobelOutputImage, width, height);
}

void copySobelConstantsToDevice(const int* hostGx, const int* hostGy)
{
	cudaMemcpyToSymbol(sobelGx, hostGx, 9 * sizeof(int));
	cudaMemcpyToSymbol(sobelGy, hostGy, 9 * sizeof(int));
}