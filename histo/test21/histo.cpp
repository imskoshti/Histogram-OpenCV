#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream> 
#include <stdio.h> 
 
using namespace std;
using namespace std;
void create_histogram_image(IplImage*, IplImage*);
 
int main()
{
    // Set up images
    IplImage *img = cvLoadImage("sanjay1.jpg", 0);
    IplImage* out = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
    // create new image structure to hold histogram image
    IplImage *hist1 = cvCreateImage(cvSize(640,480), 8, 1);
    cvSet( hist1, cvScalarAll(255), 0 );
    IplImage *hist2 = cvCreateImage(cvSize(640,480), 8, 1);
    cvSet( hist2, cvScalarAll(255), 0 );
 
       create_histogram_image(img, hist1);
       cvEqualizeHist( img, out );// Perform histogram equalization
       create_histogram_image(out, hist2);
 
    cvNamedWindow( "Original", 1) ;// Show original
    cvShowImage( "Original", img );
    cvNamedWindow( "Histogram before equalization", 1) ;//Show Histograms
    cvShowImage( "Histogram before equalization", hist1 );
    cvNamedWindow( "Histogram after equalization", 1) ;
    cvShowImage( "Histogram after equalization", hist2 );
    cvNamedWindow("Result", 1) ; // Show histogram equalized
    cvShowImage("Result", out );
    cvWaitKey(0);
 
    cvReleaseImage( &img );
    cvReleaseImage( &out );
    return 0;
}
 
 
/*******This function create histogram of the source image*******/
 
 
void create_histogram_image(IplImage* gray_img, IplImage* hist_img) {
  CvHistogram *hist;
  int hist_size = 256;
  float range[]={0,256};
  float* ranges[] = { range };
  float max_value = 0.0;
  float w_scale = 0.0000000;
  
 
hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);// create array to hold histogram values
 
cvCalcHist( &gray_img, hist, 0, NULL );// calculate histogram values
 
cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 ); // Get the minimum and maximum values of the histogram
 
cvScale( hist->bins, hist->bins, ((float)hist_img->height)/max_value, 0 );// set height by using maximim value
 
w_scale = ((float)hist_img->width)/hist_size;// calculate width
 
 
// plot the histogram
 for( int i = 0; i < hist_size; i++ ) {
 cvRectangle( hist_img, cvPoint((int)i*w_scale , hist_img->height),
 cvPoint((int)(i+1)*w_scale, hist_img->height -    
 
 cvRound(cvGetReal1D(hist->bins,i))),
 cvScalar(0), -1, 8, 0 );
    }
}
