// OpenCV Blur Image. talkera.org/opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
 
using namespace cv;
 
int main( int argc, char** argv )
{
 char* imageName = argv[1];
 
 Mat image;
 
 if( argc != 2 )
 {
   printf( " No image data. Please add filename: ./image data/opencv.png \n" );
   return -1;
 }
 image = imread( imageName, 1 );
 blur(image,image,Size(10,10));
 namedWindow( imageName, CV_WINDOW_AUTOSIZE );
 imshow( imageName, image );
 waitKey(0);
 
 return 0;
}
