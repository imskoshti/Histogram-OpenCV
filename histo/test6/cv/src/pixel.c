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
   printf( " No filename specified. \n" );
   return -1;
 }
 
 // Pixel access
 uchar pixValue;
 image = imread( imageName, 1 );

 for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
        Vec3b &intensity = image.at<Vec3b>(j, i);
        for(int k = 0; k < image.channels(); k++) {
            // calculate pixValue
            image.at<Vec3b>(j, i)[0] = 2*image.at<Vec3b>(j, i)[0];
            image.at<Vec3b>(j, i)[1] = 2*image.at<Vec3b>(j, i)[1];
            image.at<Vec3b>(j, i)[2] = 2*image.at<Vec3b>(j, i)[2];
        }
     }
 }
 
 namedWindow( "tutorialsplay.com/opencv", CV_WINDOW_AUTOSIZE );
 imshow( "tutorialsplay.com/opencv", image );
 waitKey(0);
 
 return 0;
}
