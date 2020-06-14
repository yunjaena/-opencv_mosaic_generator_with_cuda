
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\common\inc\helper_cuda.h"

#include <device_functions.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <omp.h>
#include <iostream>
#include "DS_definitions.h"
#include "DS_timer.h"

using namespace cv;
using namespace std;

__global__ void mosaic_cuda(unsigned char* in, unsigned char* out, int mon, int height, int width) {



    // cv::cuda::GpuMat image(*img);
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;


    if (ix % mon != 0 || iy % mon != 0) return;

    if (ix >= height || iy >= width) {

        return;
    }
    int R, G, B;

    //R = in[idx * 3 + 0];
    //G = in[idx * 3 + 1];
    //B = in[idx * 3 + 2];


    int nCount;
    int monWidth = mon, monHeight = mon;



    nCount = 0;
    B = 0; G = 0; R = 0;

    if ((ix + mon) > height)
        monHeight = height % mon;
    else
        monHeight = mon;

    if ((iy + mon) > width)
        monWidth = width % mon;
    else
        monWidth = mon;

    for (int n = 0; n < monHeight; n++) {
        for (int m = 0; m < monWidth; m++) {
            int idx = (ix + n) * width + (iy + m);
            B += in[idx * 3 + 0];
            G += in[idx * 3 + 1];
            R += in[idx * 3 + 2];
            //nCount++;
        }
    }

    nCount = monHeight * monWidth;
    //평균을구함

    B /= nCount;
    G /= nCount;
    R /= nCount;

    //B = 125;
    //G = 125;
    //R = 125;


    for (int n = 0; n < monHeight; n++) {
        for (int m = 0; m < monWidth; m++) {
            int idx = (ix + n) * width + (iy + m);

            out[idx * 3 + 0] = B;
            out[idx * 3 + 1] = G;
            out[idx * 3 + 2] = R;
        }
    }

}

__global__ void mosaic_cuda_sm(unsigned char* in, unsigned char* out, int mon, int height, int width) {



    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;


    if (ix % mon != 0 || iy % mon != 0) return;

    if (ix >= height || iy >= width) {

        return;
    }
    __shared__ int R;
    __shared__ int G;
    __shared__ int B;

    int nCount;
    int monWidth = mon, monHeight = mon;



    nCount = 0;
    B = 0; G = 0; R = 0;
    __syncthreads();

    if ((ix + mon) > height)
        monHeight = height % mon;
    else
        monHeight = mon;

    if ((iy + mon) > width)
        monWidth = width % mon;
    else
        monWidth = mon;

    for (int n = 0; n < monHeight; n++) {
        for (int m = 0; m < monWidth; m++) {
            int idx = (ix + n) * width + (iy + m);
            B += in[idx * 3 + 0];
            G += in[idx * 3 + 1];
            R += in[idx * 3 + 2];
        }
    }

    nCount = monHeight * monWidth;

    B /= nCount;
    G /= nCount;
    R /= nCount;

    for (int n = 0; n < monHeight; n++) {
        for (int m = 0; m < monWidth; m++) {
            int idx = (ix + n) * width + (iy + m);

            out[idx * 3 + 0] = B;
            out[idx * 3 + 1] = G;
            out[idx * 3 + 2] = R;
        }
    }

}



void mosaic_serial(const Mat* image, Mat* dst, int mon)
{
    int i, j, n, m;

    int nCount;
    int monWidth = mon, monHeight = mon;

    int R, G, B;
    for (i = 0; i < image->size().height; i += mon) {
        for (j = 0; j < image->size().width; j += mon) {
            nCount = 0;
            B = 0; G = 0; R = 0;

            if ((i + mon) > image->size().height)
                monHeight = image->size().height % mon;
            else
                monHeight = mon;

            if ((j + mon) > image->size().width)
                monWidth = image->size().width % mon;
            else
                monWidth = mon;

            for (n = 0; n < monHeight; n++) {
                for (m = 0; m < monWidth; m++) {
                    B += image->at<Vec3b>(i + n, j + m)[0];
                    G += image->at<Vec3b>(i + n, j + m)[1];
                    R += image->at<Vec3b>(i + n, j + m)[2];
                    nCount++;
                }
            }

            //평균을구함
            B /= nCount;
            G /= nCount;
            R /= nCount;

            for (n = 0; n < monHeight; n++) {
                for (m = 0; m < monWidth; m++) {
                    dst->at<Vec3b>(i + n, j + m)[0] = B;
                    dst->at<Vec3b>(i + n, j + m)[1] = G;
                    dst->at<Vec3b>(i + n, j + m)[2] = R;
                }
            }
        }
    }
}

void mosaic_openmp(const Mat* image, Mat* dst, int mon)
{
    const int NUM_THREADS = 6;
    for (int i = 0; i < image->size().height; i += mon) {
#pragma omp parallel for num_threads(NUM_THREADS)
        for (int j = 0; j < image->size().width; j += mon) {
            int B = 0, G = 0, R = 0;
            int monWidth, monHeight;

            if ((i + mon) > image->size().height)
                monHeight = image->size().height % mon;
            else
                monHeight = mon;

            if ((j + mon) > image->size().width)
                monWidth = image->size().width % mon;
            else
                monWidth = mon;

            for (int n = 0; n < monHeight; n++) {
                for (int m = 0; m < monWidth; m++) {
                    B += image->at<Vec3b>(i + n, j + m)[0];
                    G += image->at<Vec3b>(i + n, j + m)[1];
                    R += image->at<Vec3b>(i + n, j + m)[2];
                }
            }

            //평균을구함
            int nCount = monHeight * monWidth;
            B /= nCount;
            G /= nCount;
            R /= nCount;

            for (int n = 0; n < monHeight; n++) {
                for (int m = 0; m < monWidth; m++) {
                    dst->at<Vec3b>(i + n, j + m)[0] = B;
                    dst->at<Vec3b>(i + n, j + m)[1] = G;
                    dst->at<Vec3b>(i + n, j + m)[2] = R;
                }
            }
        }
    }
}

#define mosaic 40

int main()
{


    string fileName;
    Mat original_image, result_serial, result_openmp;
    Mat* original_image_cuda;
    DS_timer timer(4);
    timer.setTimerName(0, (char*)"Serial");
    timer.setTimerName(1, (char*)"Openmp");
    timer.setTimerName(2, (char*)"CUDA-normal");
    timer.setTimerName(3, (char*)"CUDA-shared memory");



    cout << "이미지 파일명을 입력해주세요 : ";
    cin >> fileName;
    original_image = imread(fileName, IMREAD_COLOR);
    if (original_image.data == NULL) {
        cout << "이미지를 찾지 못했습니다.";
        getchar();
        getchar();
        return -1;
    }

    unsigned char* input = (unsigned char*)(original_image.data);
    unsigned char* dev_input, * dev_output;
    unsigned char* output = (unsigned char*)malloc(original_image.cols * original_image.rows * 3 * sizeof(char));
    unsigned char* output_sm = (unsigned char*)malloc(original_image.cols * original_image.rows * 3 * sizeof(char));




    cudaMalloc(&original_image_cuda, sizeof(Mat));

    result_serial = Mat::zeros(original_image.size(), original_image.type());

    result_openmp = Mat::zeros(original_image.size(), original_image.type());

    timer.onTimer(0);
    mosaic_serial(&original_image, &result_serial, mosaic);
    timer.offTimer(0);

    timer.onTimer(1);
    mosaic_openmp(&original_image, &result_openmp, mosaic);
    timer.offTimer(1);




    int imgh = original_image.size().height;
    int imgw = original_image.size().width;

    int size = original_image.size().height * original_image.size().width;

    dim3 dimBlock(32, 16);
    dim3 dimGrid(ceil((float)imgh / dimBlock.x), ceil((float)imgw / dimBlock.y), 1);





    timer.onTimer(2);
    cudaMalloc((void**)&dev_input, original_image.cols * original_image.rows * 3 * sizeof(char));
    cudaMalloc((void**)&dev_output, original_image.cols * original_image.rows * 3 * sizeof(char));
    cudaMemcpy(dev_input, input, original_image.cols * original_image.rows * 3 * sizeof(char), cudaMemcpyHostToDevice);
    
    mosaic_cuda << <dimGrid, dimBlock >> > (dev_input, dev_output, mosaic, imgh, imgw);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, dev_output, original_image.cols * original_image.rows * 3 * sizeof(char), cudaMemcpyDeviceToHost);
    timer.offTimer(2);




    Mat file3 = Mat(original_image.rows, original_image.cols, original_image.type(), output);

    //------------------------------


    timer.onTimer(3);
    cudaMalloc((void**)&dev_input, original_image.cols * original_image.rows * 3 * sizeof(char));
    cudaMalloc((void**)&dev_output, original_image.cols * original_image.rows * 3 * sizeof(char));
    cudaMemcpy(dev_input, input, original_image.cols * original_image.rows * 3 * sizeof(char), cudaMemcpyHostToDevice);
   
    mosaic_cuda_sm << <dimGrid, dimBlock >> > (dev_input, dev_output, mosaic, imgh, imgw);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_sm, dev_output, original_image.cols * original_image.rows * 3 * sizeof(char), cudaMemcpyDeviceToHost);
    timer.offTimer(3);


    Mat file4 = Mat(original_image.rows, original_image.cols, original_image.type(), output_sm);




    /* show image */
    namedWindow("serial", WINDOW_NORMAL);
    imshow("serial", result_serial);

    namedWindow("openmp", WINDOW_NORMAL);
    imshow("openmp", result_openmp);

    namedWindow("CUDA-normal", WINDOW_NORMAL);
    imshow("CUDA-normal", file3);

    namedWindow("CUDA-shared memory", WINDOW_NORMAL);
    imshow("CUDA-shared memory", file4);


    timer.printTimer();
    waitKey();
    destroyAllWindows();





}