
// 用于跟踪并画框

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <tinyxml.h>

using namespace std;
using namespace cv;


// 下面需要改动！！！！！！！！！！！！！！！！！！！！！！！！！！
bool use_default_video_adress = false;
string video_adress = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_videos/turtlebot2_1.mp4";	//设定默认视频地址，也可以在参数里改
bool save_image = true;	//默认不储存
bool use_default_save_root = false;
string save_root = "/home/zq610/WYZ/tiny_code/dataset_handle/output/";	//默认存储位置
// 上面需要改动！！！！！！！！！！！！！！！！！！！！！！！！！！！！

Rect2d box;//矩形对象
bool drawing_box = false;//记录是否在画矩形对象
bool gotBB = false;
int mode = 0;	//模式选择,默认0打开摄像头，1从视频读取
int camera_choose = 0;	//选择摄像头，默认0
bool show_image = true;


string int_to_string(int input_int){
	ostringstream stream;
	stream << input_int;
	return stream.str();
}

int string_to_int(string intput_string){
	int output_int;
	stringstream ss;
	ss << intput_string;
	ss >> output_int;
	if (!ss.good()){
		printf("There is an error in transferring\n");
		exit(1);
	}
	return output_int;
}

//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
    case CV_EVENT_LBUTTONDOWN://鼠标左键按下事件
	    drawing_box = true;//标志在画框
	    box = Rect2d( x, y, 0, 0 );//记录矩形的开始的点
    	break;
	case CV_EVENT_MOUSEMOVE://鼠标移动事件
		if (drawing_box){//如果左键一直按着，则表明在画矩形
			box.width = x-box.x;
			box.height = y-box.y;//更新长宽
		}
		break;
	case CV_EVENT_LBUTTONUP://鼠标左键松开事件
		drawing_box = false;//不在画矩形
		if( box.width < 0 ){//排除宽为负的情况，在这里判断是为了优化计算，不用再移动时每次更新都要计算长宽的绝对值
			box.x += box.width;//更新原点位置，使之始终符合左上角为原点
			box.width *= -1;//宽度取正
		}
		if( box.height < 0 ){//同上
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	default:
		break;
  }
}


void read_options(int argc, char** argv){
	for (int i = 0; i < argc; ++i)
	{
		if(strcmp(argv[i], "-v") == 0){	//从视频流读取
            printf("read from a video\n");
			mode = 1;
			if(!use_default_video_adress && argc>i){
				video_adress = argv[i+1];	//视频位置
			}
		}
		if(strcmp(argv[i], "-c") == 0){	//从摄像头读取
            printf("read from the camera\n");
			mode = 0;
			if(argc>i){
				camera_choose = string_to_int(argv[i+1]);	//选择摄像头，默认0
			}
		}
		if(strcmp(argv[i], "-s") == 0){	//存储目录
            printf("change the default saving direction\n");
			save_image = true;
			if(argc>i && !use_default_save_root){
				printf("%s\n", argv[i+1]);
				save_root = argv[i+1];
			}
		}
		if(strcmp(argv[i], "-noshow") == 0){
            printf("not show the video\n");
			show_image = false;
		}
	}
	printf("after reading argument\n");
}



int main(int argc, char** argv){
	// can change to BOOSTING, MIL, KCF (OpenCV 3.1), TLD, MEDIANFLOW, or GOTURN (OpenCV 3.2)
	// Ptr<Tracker> tracker = Tracker::create("KCF"); 
	//TrackerKCF::Params::read("/usr/q.txt");
	Ptr<TrackerKCF> tracker = TrackerKCF::createTracker();
	read_options(argc, argv);

	VideoCapture video;
	if (mode == 0){
		VideoCapture video(0);	//用电脑自带摄像头比外置摄像头快很多
	}
	else if (mode == 1){
		video.open(video_adress);
	}

	if(!video.isOpened()){
		cerr << "cannot read video!" << endl;
		return -1;
	}
	Mat frame;
	Mat temp;
	video.read(frame);
	// namedWindow("Tracking");
	namedWindow("Tracking", 0);//WINDOW_NORMAL=0，在这个模式下可以调整窗口的大小。
	cvResizeWindow("Tracking", 1920, 1080);
	//cvResizeWindow("Tracking", 720, 480);
	setMouseCallback("Tracking", mouseHandler, &frame);
	imshow("Tracking", frame);

	while(!gotBB){
		//只要不再次按下鼠标左键触发事件,则程序显示的一直是if条件里面被矩形函数处理过的temp图像，如果再次按下鼠标左键就进入if，不断更新被画矩形函数处理过的temp，因为处理速度快所以看起来画矩形的过程是连续的没有卡顿，因为每次重新画都是在没有框的基础上画出新的框因为人眼的残影延迟所以不会有拖影现象。每次更新矩形框的传入数据是重新被img（没有框）的数据覆盖的temp（即img.data==temp.data）和通过回调函数更新了的Box记录的坐标点数据。
		waitKey(1);//维持imshow
		if(drawing_box){//不断更新正在画的矩形
			frame.copyTo(temp);//这句放在这里是保证了每次更新矩形框都是在没有原图的基础上更新矩形框。
			rectangle(temp, box, Scalar(255, 255, 255), 2, 1);
			imshow("Tracking", temp);//显示
		}
	}

	//Rect2d box(270, 120, 180, 260);
	tracker->init(frame, box);
	int num = 1;
	while(video.read(frame)){
		tracker->update(frame, box);
		
		// if roi-box cross the border, and handle this
		if (!(0<=box.x && box.x+box.width<=frame.cols && 0<=box.y && box.y+box.height<=frame.rows)){
			printf("There is somewhere cross the border!\n");
			if(0 > box.x){
				box.x = 0;
			}
			if(box.x+box.width > frame.cols){
				box.width = frame.cols - box.x;
			}
			if(0 > box.y){
				box.y = 0;
			}
			if(box.y+box.height > frame.rows){
				box.height = frame.rows - box.y;
			}
    	}
		
		if(show_image){
			rectangle(frame, box, Scalar(255, 0, 0), 2, 1);//如果存储图片就不画框
			imshow("Tracking", frame);
			int k=waitKey(1);
			if(k==27) break;
		}
		if (save_image)
		{
			Mat frame_roi = frame(box);
			string file_name = save_root + "/" + int_to_string(num) + ".jpg";
			imwrite(file_name, frame_roi);
			if (num % 20 == 0)
				printf("have handled %d frames\n", num);
			num ++;
		}
	}
}