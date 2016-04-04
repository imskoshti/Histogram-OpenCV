//#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"using namespace cv;
#include <iostream>using namespace std;

int main()
{
Mat img = imread("sanjay.jpg");
Mat imgH;
Mat imgL;
imgH = img + Scalar(100, 100, 100);
imgL = img + Scalar(-100, -100, -100);

namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
namedWindow("Increased Brightness Image", CV_WINDOW_AUTOSIZE);
namedWindow("Decreased Brightness Image", CV_WINDOW_AUTOSIZE);

imshow("Original Image", img);
imshow("Increased Brightness Image", imgH);
imshow("Decreased Brightness Image", imgL);
cvWaitKey(0);

return 0;
}
