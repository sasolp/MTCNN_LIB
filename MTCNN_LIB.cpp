#include "stdafx.h"
#include <stdio.h>
#include "MTCNN_LIB.h"

#include "net.h"
bool cmpScore(orderScore lsh, orderScore rsh){
	if(lsh.score<rsh.score)
		return true;
	else
		return false;
}


class mtcnn{
public:
	mtcnn();
	void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
	bool loaded;
	int err_code;
	int minsize;
	float nms_threshold[3] ;
	float threshold[3] ;
	int facialKeisDetection;
private:
	void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale);
	void nms(vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname="Union");
	void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);

	ncnn::Net Pnet, Rnet, Onet;
	ncnn::Mat img;
	float mean_vals[3] ;
	float norm_vals[3] ;
	std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
	std::vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
	int img_w, img_h;
};

mtcnn::mtcnn(){
	loaded = true;
	err_code = 0;
	minsize = 40;
	nms_threshold[0] = 0.5f;nms_threshold[1] =  0.7f; nms_threshold[2] =  0.7f;
	threshold[0] = 0.5f; threshold[1] = 0.4f; threshold[2] = 0.4f;
	mean_vals[0] = 127.5f; mean_vals[1] = 127.5f; mean_vals[2] = 127.5f;
	norm_vals[0] = 0.0078125f; norm_vals[1] = 0.0078125f; norm_vals[2] = 0.0078125f;
	int succeed = Pnet.load_param("det1.param");
	if(succeed)
	{
		err_code = -1;
		loaded = false;
	}
	else
	{
		succeed = Pnet.load_model("det1.bin");
		if(succeed)
		{
			err_code = -2;
			loaded = false;
		}
		else
		{
			succeed = Rnet.load_param("det2.param");
			if(succeed)
			{
				err_code = -3;
				loaded = false;
			}
			else
			{
				succeed = Rnet.load_model("det2.bin");
				if(succeed)
				{
					err_code = -4;
					loaded = false;
				}
				else
				{
					succeed = Onet.load_param("det3.param");
					if(succeed)
					{
						err_code = -5;
						loaded = false;
					}
					else
					{
						succeed = Onet.load_model("det3.bin");
						if(succeed)
						{
							err_code = -6;
							loaded = false;
						}
					}
				}
			}
		}
	}
}
double round(double number)
{
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}
void mtcnn::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale){
	int stride = 2;
	int cellsize = 12;
	int count = 0;
	//score p
	float *p = score.channel(1);//score.data + score.cstep;
	float *plocal = location.data;
	Bbox bbox;
	orderScore order;
	for(int row=0;row<score.h;row++){
		for(int col=0;col<score.w;col++){
			if(*p>threshold[0]){
				bbox.score = *p;
				order.score = *p;
				order.oriOrder = count;
				bbox.x1 = (int)round((stride*col+1)/scale);
				bbox.y1 = (int)round((stride*row+1)/scale);
				bbox.x2 = (int)round((stride*col+1+cellsize)/scale);
				bbox.y2 = (int)round((stride*row+1+cellsize)/scale);
				bbox.exist = true;
				bbox.area = float(bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
				for(int channel=0;channel<4;channel++)
					bbox.regreCoord[channel]=location.channel(channel)[0];
				boundingBox_.push_back(bbox);
				bboxScore_.push_back(order);
				count++;
			}
			p++;
			plocal++;
		}
	}
}
void mtcnn::nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname){
	if(boundingBox_.empty()){
		return;
	}
	std::vector<int> heros;
	//sort the score
	sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

	int order = 0;
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	while(bboxScore_.size()>0){
		order = bboxScore_.back().oriOrder;
		bboxScore_.pop_back();
		if(order<0)continue;
		if(boundingBox_.at(order).exist == false) continue;
		heros.push_back(order);
		boundingBox_.at(order).exist = false;//delete it

		for(int num=0;num<boundingBox_.size();num++){
			if(boundingBox_.at(num).exist){
				//the iou
				maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?(float)boundingBox_.at(num).x1:(float)boundingBox_.at(order).x1;
				maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?(float)boundingBox_.at(num).y1:(float)boundingBox_.at(order).y1;
				minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?(float)boundingBox_.at(num).x2:(float)boundingBox_.at(order).x2;
				minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?(float)boundingBox_.at(num).y2:(float)boundingBox_.at(order).y2;
				//maxX1 and maxY1 reuse 
				maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
				maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
				//IOU reuse for the area of two bbox
				IOU = maxX * maxY;
				if(!modelname.compare("Union"))
					IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
				else if(!modelname.compare("Min")){
					IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
				}
				if(IOU>overlap_threshold){
					boundingBox_.at(num).exist=false;
					for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
						if((*it).oriOrder == num) {
							(*it).oriOrder = -1;
							break;
						}
					}
				}
			}
		}
	}
	for(int i=0;i<heros.size();i++)
		boundingBox_.at(heros.at(i)).exist = true;
}
void mtcnn::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width){
	if(vecBbox.empty()){
		cout<<"Bbox is empty!!"<<endl;
		return;
	}
	float bbw=0, bbh=0, maxSide=0;
	float h = 0, w = 0;
	float x1=0, y1=0, x2=0, y2=0;
	for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
		if((*it).exist){
			bbw = (*it).x2 - (*it).x1 + 1.0f;
			bbh = (*it).y2 - (*it).y1 + 1.0f;
			x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
			y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
			x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
			y2 = (*it).y2 + (*it).regreCoord[3]*bbh;

			w = x2 - x1 + 1;
			h = y2 - y1 + 1;

			maxSide = (h>w)?h:w;
			x1 = x1 + w*0.5f - maxSide*0.5f;
			y1 = y1 + h*0.5f - maxSide*0.5f;
			(*it).x2 = (int)round(x1 + maxSide - 1);
			(*it).y2 = (int)round(y1 + maxSide - 1);
			(*it).x1 = (int)round(x1);
			(*it).y1 = (int)round(y1);

			//boundary check
			if((*it).x1<0)(*it).x1=0;
			if((*it).y1<0)(*it).y1=0;
			if((*it).x2>width)(*it).x2 = width - 1;
			if((*it).y2>height)(*it).y2 = height - 1;

			it->area = float(it->x2 - it->x1)*(it->y2 - it->y1);
		}
	}
}
void mtcnn::detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox_){
	firstBbox_.clear();
	firstOrderScore_.clear();
	secondBbox_.clear();
	secondBboxScore_.clear();
	thirdBbox_.clear();
	thirdBboxScore_.clear();

	img = img_;
	img_w = img.w;
	img_h = img.h;
	img.substract_mean_normalize(mean_vals, norm_vals);

	float minl = img_w<img_h?(float)img_w:(float)img_h;
	int MIN_DET_SIZE = 12;

	float m = (float)MIN_DET_SIZE/minsize;
	minl *= m;
	float factor = 0.709f;
	int factor_count = 0;
	vector<float> scales_;
	while(minl>MIN_DET_SIZE){
		if(factor_count>0)m = m*factor;
		scales_.push_back(m);
		minl *= factor;
		factor_count++;
	}
	orderScore order;
	int count = 0;

	for (size_t i = 0; i < scales_.size(); i++) {
		int hs = (int)ceil(img_h*scales_[i]);
		int ws = (int)ceil(img_w*scales_[i]);
		//ncnn::Mat in = ncnn::Mat::from_pixels_resize(image_data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, ws, hs);
		ncnn::Mat in;
		resize_bilinear(img_, in, ws, hs);
		//in.substract_mean_normalize(mean_vals, norm_vals);
		ncnn::Extractor ex = Pnet.create_extractor();
		ex.set_light_mode(true);
		ex.input("data", in);
		ncnn::Mat score_, location_;
		ex.extract("prob1", score_);
		ex.extract("conv4-2", location_);
		std::vector<Bbox> boundingBox_;
		std::vector<orderScore> bboxScore_;
		generateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
		nms(boundingBox_, bboxScore_, nms_threshold[0]);

		for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++){
			if((*it).exist){
				firstBbox_.push_back(*it);
				order.score = (*it).score;
				order.oriOrder = count;
				firstOrderScore_.push_back(order);
				count++;
			}
		}
		bboxScore_.clear();
		boundingBox_.clear();
	}
	//the first stage's nms
	if(count<1)return;
	nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
	refineAndSquareBbox(firstBbox_, img_h, img_w);
	printf("firstBbox_.size()=%d\n", firstBbox_.size());

	//second stage
	count = 0;
	for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
		if((*it).exist){
			ncnn::Mat tempIm;
			copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
			ncnn::Mat in;
			resize_bilinear(tempIm, in, 24, 24);
			ncnn::Extractor ex = Rnet.create_extractor();
			ex.set_light_mode(true);
			ex.input("data", in);
			ncnn::Mat score, bbox;
			ex.extract("prob1", score);
			ex.extract("conv5-2", bbox);
			if(*(score.data+score.cstep)>threshold[1]){
				for(int channel=0;channel<4;channel++)
					it->regreCoord[channel]=bbox.channel(channel)[0];//*(bbox.data+channel*bbox.cstep);
				it->area = float(it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = score.channel(1)[0];//*(score.data+score.cstep);
				secondBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				secondBboxScore_.push_back(order);
			}
			else{
				(*it).exist=false;
			}
		}
	}
	printf("secondBbox_.size()=%d\n", secondBbox_.size());
	if(count<1)return;
	nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
	refineAndSquareBbox(secondBbox_, img_h, img_w);

	//third stage 
	count = 0;
	for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
		if((*it).exist){
			ncnn::Mat tempIm;
			copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
			ncnn::Mat in;
			resize_bilinear(tempIm, in, 48, 48);
			ncnn::Extractor ex = Onet.create_extractor();
			ex.set_light_mode(true);
			ex.input("data", in);
			ncnn::Mat score, bbox, keyPoint;
			ex.extract("prob1", score);
			ex.extract("conv6-2", bbox);
			ex.extract("conv6-3", keyPoint);
			if(score.channel(1)[0]>threshold[2]){
				for(int channel=0;channel<4;channel++)
					it->regreCoord[channel]=bbox.channel(channel)[0];
				it->area = float(it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = score.channel(1)[0];
				for(int num=0;num<5;num++){
					(it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(num)[0];
					(it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(num+5)[0];
				}

				thirdBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				thirdBboxScore_.push_back(order);
			}
			else
				(*it).exist=false;
		}
	}

	printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
	if(count<1)return;
	refineAndSquareBbox(thirdBbox_, img_h, img_w);
	nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
	finalBbox_ = thirdBbox_;
}

int main(int argc, char** argv)
{
	const char* imagepath = argv[1];

	cv::Mat cv_img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
	if (cv_img.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	}
	std::vector<Bbox> finalBbox;
	mtcnn mm;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);

	mm.detect(ncnn_img, finalBbox);

	for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++){
		if((*it).exist){

			cv::rectangle(cv_img, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0,0,255), 2,8,0);
			for(int num=0;num<5;num++)cv::circle(cv_img,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
		}
	}
	imwrite("result.jpg",cv_img);
	return 0;
}

extern "C" __declspec(dllexport) int __stdcall CreateFaceNetObj(char* model_deploy, char* model_weights, void** net_out)
{
	int ret_val = 0;
	//static dnn::Net net;
	//try
	//{
	mtcnn *mtcnn_ptr = new mtcnn();
	////}
	////catch(cv::Exception &ex)
	////{
	////	cout << ex.msg;
	////}
	if (mtcnn_ptr->err_code)
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "cfg-file:     " << model_deploy << endl;
		cerr << "weights-file: " << model_weights << endl;
		cerr << "Models can be downloaded here:" << endl;
		cerr << "https://pjreddie.com/darknet/yolo/" << endl;
		ret_val = -1;
	}
	else
	{
		*net_out = mtcnn_ptr;
	}
	return ret_val;
}
extern "C" __declspec(dllexport) int __stdcall ReleaseFaceNetObj(void** mtcnn_ptr)
{
	int ret_val = 0;
	mtcnn *mm = (mtcnn*)*mtcnn_ptr;
	if(mm)
		delete mm;
	return ret_val;
}
void read_from_data(cv::Mat &dst, uchar* image_data, int width, int height, int channels, int stride)
{

	if(channels > 1)
	{
		dst = Mat(height, width, CV_8UC3);
		uchar* dst_data = dst.data;
		for(int i = 0 ; i < height; i++)
		{
			memcpy(dst_data, image_data, width * channels);
			image_data += stride;
			dst_data += width * channels;
		}
	}
	else
	{
		dst = Mat(height, width, CV_8UC1);
		uchar* dst_data = dst.data;
		for(int i = 0 ; i < height; i++)
		{
			memcpy(dst_data, image_data, width * channels);
			image_data += stride;
			dst_data += width * channels;
		}
		cv::cvtColor(dst, dst, CV_GRAY2BGR);
	}
}
extern "C" __declspec(dllexport) int __stdcall ReleaseObjects(void** objs)
{
	DetectedFace* objects = (DetectedFace*)(*objs);
	if(objects)
		delete[] objects;
	return 0;
}
extern "C" __declspec(dllexport) int __stdcall ExtractFaces(void** mtcnn_ptr, uchar* image_data, int width, int height, int channels, int stride, float min_confidence, DetectedFace** objects, int* detected_objects, UserFaceSettings face_settings)
{
	int ret_val = 0;
	mtcnn *mm = ((mtcnn*)*mtcnn_ptr);
	//char* path = (char*)image_data;
	//cv::Mat cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
	// if (cv_img.empty())
	//   {
	//       fprintf(stderr, "cv::imread %s failed\n", path);
	//       return -1;
	//   }
	cv::Mat cv_img;
	read_from_data(cv_img, image_data, width, height, channels, stride);
	std::vector<Bbox> finalBbox;

	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);
	mm->minsize = face_settings.sizeThresh;
	mm->facialKeisDetection = face_settings.facialKeisDetection;
	mm->nms_threshold[0] = 0.5;mm->nms_threshold[1] =  face_settings.nmsThresh; mm->nms_threshold[2] =  face_settings.nmsThresh;
	mm->threshold[0] = face_settings.pThresh; mm->threshold[1] = face_settings.rThresh; mm->threshold[2] = face_settings.rThresh;
	mm->detect(ncnn_img, finalBbox);
	*detected_objects = (int)finalBbox.size();

	if(detected_objects[0] > 0)
	{
		*objects = new DetectedFace[*detected_objects];
		int obj_counter = 0;
		for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++){
			if((*it).exist){
				Bbox box = (*it);
				DetectedFace new_obj;
				new_obj.x = box.x1;
				new_obj.y = box.y1;
				new_obj.width = box.x2 - box.x1;
				new_obj.height = box.y2 - box.y1;
				for(int num=0;num<5;num++)
				{
					Point pt = Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5));
					new_obj.pnts[num].x =  pt.x;
					new_obj.pnts[num].y =  pt.y;
				}
				//rectangle(cv_img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(255,0,0), 2);
				(*objects)[obj_counter++] = new_obj;
			}
		}
		*detected_objects = obj_counter;
	}
	return 0;
}