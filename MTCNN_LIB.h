#pragma once
#include <algorithm>
#include <vector>
#include <math.h>
#include <iostream>
#include <cv.h>
#include <opencv2\imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "resource.h"
using namespace std;
using namespace cv;
typedef struct _Point
{
	int x;
	int y;
}_Point;
typedef struct _DetectedFace
{
	int x;
	int y;
	int width;
	int height;
	int id;
	float confidenc;
	_Point pnts[5];
}DetectedFace;
typedef struct _UserFaceSettings
{
	float pThresh;
	float rThresh;
	float nmsThresh;
	int sizeThresh;
	int facialKeisDetection;
}UserFaceSettings;
struct Bbox
{
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	bool exist;
	float ppoint[10];
	float regreCoord[4];
};

struct orderScore
{
	float score;
	int oriOrder;
};
extern "C" __declspec(dllexport) int __stdcall CreateFaceNetObj(char* model_deploy, char* model_weights, void** net_out);
extern "C" __declspec(dllexport) int __stdcall ReleaseFaceNetObj(void** mtcnn_ptr);
void read_from_data(cv::Mat &dst, uchar* image_data, int width, int height, int channels, int stride);
extern "C" __declspec(dllexport) int __stdcall ReleaseObjects(void** objs);
extern "C" __declspec(dllexport) int __stdcall ExtractFaces(void** mtcnn_ptr, uchar* image_data, int width, int height, int channels, int stride, float min_confidence, DetectedFace** objects, int* detected_objects, UserFaceSettings face_settings);