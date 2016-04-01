#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
 
using namespace cv;
using namespace std;
 
/// Global Variables
Mat img; Mat templ; Mat result;
const char* image_window = "Source Image";
const char* result_window = "Result window";
 
int match_method;
 
/// Function Headers
void MatchingMethod( int, void* );
 
/**
* @function main
*/
int main( int, char** argv )
{
  /// Load image and template
  img = imread( "data/where_is_wally.jpg", 1 );
  templ = imread( "data/wally.jpg", 1 );
 
  // Match
  MatchingMethod( 0, 0 );
 
  waitKey(0);
  return 0;
}
 
/**
* @function MatchingMethod
* @brief Trackbar callback
*/
void MatchingMethod( int, void* )
{
  /// Source image to display
  Mat img_display;
  img.copyTo( img_display );
 
  /// Create the result matrix
  int result_cols = img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;
 
  result.create( result_cols, result_rows, CV_32FC1 );
 
  /// Do the Matching and Normalize
  matchTemplate( img, templ, result, match_method );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
 
  /// Localizing the best match with minMaxLoc
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;
 
  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
 
 
  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }
 
  /// Show me what you got
  rectangle( img, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,0,255), 4, 8, 0 );
  imshow( result_window, img );
 
  return;
}
