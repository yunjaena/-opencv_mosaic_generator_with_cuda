//
//#include <cuda.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "device_atomic_functions.h"
//
//#include <device_functions.h>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include <omp.h>
//#include <iostream>
//#include "DS_definitions.h"
//#include "DS_timer.h"
//
//using namespace cv;
//using namespace std;
//
//void mosaic_serial(const Mat* image, Mat* dst, int mon)
//{
//    int i, j, n, m;
//
//    int nCount;
//    int monWidth = mon, monHeight = mon;
//
//    int R, G, B;
//    for (i = 0; i < image->size().height; i += mon) {
//        for (j = 0; j < image->size().width; j += mon) {
//            nCount = 0;
//            B = 0; G = 0; R = 0;
//
//            if ((i + mon) > image->size().height)
//                monHeight = image->size().height % mon;
//            else
//                monHeight = mon;
//
//            if ((j + mon) > image->size().width)
//                monWidth = image->size().width % mon;
//            else
//                monWidth = mon;
//
//            for (n = 0; n < monHeight; n++) {
//                for (m = 0; m < monWidth; m++) {
//                    B += image->at<Vec3b>(i + n, j + m)[0];
//                    G += image->at<Vec3b>(i + n, j + m)[1];
//                    R += image->at<Vec3b>(i + n, j + m)[2];
//                    nCount++;
//                }
//            }
//
//            //평균을구함
//            B /= nCount;
//            G /= nCount;
//            R /= nCount;
//
//            for (n = 0; n < monHeight; n++) {
//                for (m = 0; m < monWidth; m++) {
//                    dst->at<Vec3b>(i + n, j + m)[0] = B;
//                    dst->at<Vec3b>(i + n, j + m)[1] = G;
//                    dst->at<Vec3b>(i + n, j + m)[2] = R;
//                }
//            }
//        }
//    }
//}
//
//void mosaic_openmp(const Mat* image, Mat* dst, int mon)
//{
//    const int NUM_THREADS = 8;
//    for (int i = 0; i < image->size().height; i += mon) {
//#pragma omp parallel for num_threads(NUM_THREADS)
//        for (int j = 0; j < image->size().width; j += mon) {
//            int B = 0, G = 0, R = 0;
//            int monWidth, monHeight;
//
//            if ((i + mon) > image->size().height)
//                monHeight = image->size().height % mon;
//            else
//                monHeight = mon;
//
//            if ((j + mon) > image->size().width)
//                monWidth = image->size().width % mon;
//            else
//                monWidth = mon;
//
//            for (int n = 0; n < monHeight; n++) {
//                for (int m = 0; m < monWidth; m++) {
//                    B += image->at<Vec3b>(i + n, j + m)[0];
//                    G += image->at<Vec3b>(i + n, j + m)[1];
//                    R += image->at<Vec3b>(i + n, j + m)[2];
//                }
//            }
//
//            //평균을구함
//            int nCount = monHeight * monWidth;
//            B /= nCount;
//            G /= nCount;
//            R /= nCount;
//
//            for (int n = 0; n < monHeight; n++) {
//                for (int m = 0; m < monWidth; m++) {
//                    dst->at<Vec3b>(i + n, j + m)[0] = B;
//                    dst->at<Vec3b>(i + n, j + m)[1] = G;
//                    dst->at<Vec3b>(i + n, j + m)[2] = R;
//                }
//            }
//        }
//    }
//}
//
//int main()
//{
//    string fileName;
//    Mat original_image, result_serial, result_openmp;
//    DS_timer timer(2);
//    timer.setTimerName(0, (char*)"Serial");
//    timer.setTimerName(1, (char*)"Openmp");
//
//    cout << "이미지 파일명을 입력해주세요 : ";
//    cin >> fileName;
//    original_image = imread(fileName, IMREAD_COLOR);
//    if (original_image.data == NULL) {
//        cout << "이미지를 찾지 못했습니다.";
//        getchar();
//        getchar();
//        return -1;
//    }
//
//    result_serial = Mat::zeros(original_image.size(), original_image.type());
//    result_openmp = Mat::zeros(original_image.size(), original_image.type());
//
//    timer.onTimer(0);
//    mosaic_serial(&original_image, &result_serial, 50);
//    timer.offTimer(0);
//
//    timer.onTimer(1);
//    mosaic_openmp(&original_image, &result_openmp, 50);
//    timer.offTimer(1);
//
//    /* show image */
//    namedWindow("serial", WINDOW_NORMAL);
//    imshow("serial", result_serial);
//
//    namedWindow("openmp", WINDOW_NORMAL);
//    imshow("openmp", result_openmp);
//
//    timer.printTimer();
//    waitKey();
//    destroyAllWindows();
//}
