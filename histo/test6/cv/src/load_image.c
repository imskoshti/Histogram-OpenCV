// OpenCV Load and display image, http://tutorialsplay.com/opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
 
using namespace cv;
 
int main( int argc, char** argv )
{
 char* imageName = argv[1];
 
 Mat image;
 if( argc < 2)
 {
   printf( "Please specify filename: ./image data/opencv.png\n" );
   exit(-1);
 }

 image = imread( imageName, 1 );
 namedWindow( imageName, CV_WINDOW_AUTOSIZE );
 imshow( imageName, image );
 waitKey(0);
 
 return 0;
}
