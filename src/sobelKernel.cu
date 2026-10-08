__global__ void sobelKernel(unsigned char* dev_imageData, unsigned char* dev_sobelOutputImage, int width, int height)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx >= width || idy >= height)
	{
		return;
	}

	int offset = idx + idy * width;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int neighborX = idx + (j - 1);
			int neighborY = idy + (i - 1);

			neighbourOffset = neighborX + neighborY * width;

			RGBOffset = neighbourOffset * 3;

			int pixelValue = dev_imageData[RGBOffset];




		}
	}
}