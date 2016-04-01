#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat imHist(Mat hist, float scaleX=1, float scaleY=1){
  double maxVal=0;
  minMaxLoc(hist, 0, &maxVal, 0, 0);
  int rows = 64; //default height size
  int cols = hist.rows; //get the width size from the histogram
  Mat histImg = Mat::zeros(rows*scaleX, cols*scaleY, CV_8UC3);
  //for each bin
  for(int i=0;i<cols-1;i++) {
    float histValue = hist.at<float>(i,0);
    float nextValue = hist.at<float>(i+1,0);
    Point pt1 = Point(i*scaleX, rows*scaleY);
    Point pt2 = Point(i*scaleX+scaleX, rows*scaleY);
    Point pt3 = Point(i*scaleX+scaleX, (rows-nextValue*rows/maxVal)*scaleY);
    Point pt4 = Point(i*scaleX, (rows-nextValue*rows/maxVal)*scaleY);

    int numPts = 5;
    Point pts[] = {pt1, pt2, pt3, pt4, pt1};

    fillConvexPoly(histImg, pts, numPts, Scalar(255,255,255));
  }
  return histImg;
}

int main( int argc, char** argv ) {
  // check for supplied argument
  if( argc < 2 ) {
    cout << "Usage: loadimg <filename>\n" << endl;
    return 1;
  }

  // load the image, load the image in grayscale
  Mat img = imread( argv[1], CV_LOAD_IMAGE_COLOR );

  // always check
  if( img.data == NULL ) {
    cout << "Cannot load file " << argv[1] << endl;
    return 1;
  }

  //Hold the histogram
  MatND hist;
  Mat histImg;
  int nbins = 256; // lets hold 256 levels
  int hsize[] = { nbins }; // just one dimension
  float range[] = { 0, 255 };
  const float *ranges[] = { range };
  int chnls[] = {0};

  // create colors channels
  vector<Mat> colors;
  split(img, colors);

  // compute for all colors
  calcHist(&colors[0], 1, chnls, Mat(), hist,1,hsize,ranges);
  histImg = imHist(hist,3,3);
  imshow("Blue",histImg);

  calcHist(&colors[1], 1, chnls, Mat(), hist,1,hsize,ranges);
  histImg = imHist(hist,3,3);
  imshow("Green",histImg);

  calcHist(&colors[2], 1, chnls, Mat(), hist,1,hsize,ranges);
  histImg = imHist(hist,3,3);
  imshow("Red",histImg);

  // show image
  imshow("Image", img);

  // wait until user press a key
  waitKey(0);

  // no need to release the memory, Mat do it for you
  return 0;
}
