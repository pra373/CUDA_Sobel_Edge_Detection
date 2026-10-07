__global__ void greyScaleKernel(unsigned char* dev_imageData, unsigned char* dev_greyScaleOutputImage, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int offset = (x + y * width);
	int RGBOffset = offset * 3;

	int R = dev_imageData[RGBOffset];
	int G = dev_imageData[RGBOffset + 1];
	int B = dev_imageData[RGBOffset + 2];

	int grey = ((0.299 * R) + (0.587 * G) + (0.114 * B));

	dev_greyScaleOutputImage[RGBOffset] = grey;
	dev_greyScaleOutputImage[RGBOffset + 1] = grey;
	dev_greyScaleOutputImage[RGBOffset + 2] = grey;

}

void launchGreyScaleKernel(unsigned char* dev_imageData, unsigned char* dev_greyScaleOutputImage, int width, int height)
{
	dim3 blockDim(32, 32);
	dim3 gridDim((width + (blockDim.x - 1)) / blockDim.x, (height + (blockDim.y - 1)) / blockDim.y);

	greyScaleKernel << <gridDim, blockDim >> > (dev_imageData, dev_greyScaleOutputImage, width, height);
}