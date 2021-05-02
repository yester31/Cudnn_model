#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//이미지 열기

int main(int, char) {

	Mat img = imread("test2.jpg", 1); // 0이면 그레이 스케일

	namedWindow("imageWindow", 0); // 0 이면 resizing 가능

	imshow("imageWindow", img);

	waitKey(0);
	


	return 0;

}
