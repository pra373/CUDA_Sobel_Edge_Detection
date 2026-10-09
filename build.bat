cls

cl.exe /c /EHsc src/main.cpp /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include" /Fo:build/main.obj

nvcc.exe -c src/greyScaleKernel.cu -o build/greyScaleKernel.obj 
nvcc.exe -c src/sobelKernel.cu -o build/sobelKernel.obj 

link.exe build/main.obj build/greyScaleKernel.obj build/sobelKernel.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\lib\x64" cudart.lib /out:build/main.exe
