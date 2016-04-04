//#include <opencv2\opencv.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
using namespace std;
using namespace cv;


void drawHist(const vector<int>& data, Mat3b& dst, int binSize = 3, int height = 0)
{
    int max_value = *max_element(data.begin(), data.end());
    int rows = 0;
    int cols = 0;
    if (height == 0) {
        rows = max_value + 10;
    } else { 
        rows = max(max_value + 10, height);
    }

    cols = data.size() * binSize;

    dst = Mat3b(rows, cols, Vec3b(0,0,0));

    for (int i = 0; i < data.size(); ++i)
    {
        int h = rows - data[i];
        rectangle(dst, Point(i*binSize, h), Point((i + 1)*binSize-1, rows), (i%2) ? Scalar(0, 100, 255) : Scalar(0, 0, 255), CV_FILLED);
    }

}

int main()
{
    vector<int> hist = (10,20,12,23,25,45,6);  
    Mat3b image;
    drawHist(hist, image);

    imshow("Histogram", image);
    waitKey();

    return 0;
}
