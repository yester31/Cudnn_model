#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//�̹��� ����

int main(int, char) {

	Mat img = imread("test2.jpg", 1); // 0�̸� �׷��� ������

	namedWindow("imageWindow", 0); // 0 �̸� resizing ����

	imshow("imageWindow", img);

	waitKey(0);
	


	return 0;

}
