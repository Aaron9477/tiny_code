////////////////////////////////////////////////////
//	用于在视频流中跟踪目标物体,并且以VOC数据集格式,存储数据
//	ESC退出,跟踪失效时按space暂停,重新画取bbox
//  注意:处理多个视频时,每处理完一个视频,需要更新存储图片的顺序号
////////////////////////////////////////////////////
// 	TODO:1.数据集文件夹创建 2.设定(自动识别)视频流的分别率,注意使用的地方不止show中,还有画线中
//		 3.跟踪过程中输入跟踪物体类别
////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
// #include <iomanip>
// access函数使用的库文件,access用于检查路径是否存在
#include <unistd.h>
#include "tinyxml.h"
#include "write_xml.h"

using namespace std;
using namespace cv;



// 下面需要改动！！！！！！！！！！！！！！！！！！！！！！！！！！
bool use_default_video_adress = false;
string video_adress = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_videos/turtlebot3_12.mp4";	//设定默认视频地址，也可以在参数里改
bool save_image = true;	//默认不储存
bool use_default_save_root = false;
string save_root = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/VOC_type/turtlebot";	//默认存储位置,目录下应有Annotations和ImageSets
int interval = 15;	// the interval between two saved picture
string track_object_type = "car";
int saved_pic_start = 1097;
// 上面需要改动！！！！！！！！！！！！！！！！！！！！！！！！！！！！

Rect2d box;	// 矩形对象,全局作用
bool drawing_box = false;//记录是否在画矩形对象
bool gotBB = false;
int mode = 1;	//模式选择,0打开摄像头，1从视频读取
int camera_choose = 0;	//选择摄像头，默认0
bool show_image = true;	// whether show the tracking process

// string int_to_string(int input_int){
// 	ostringstream stream;
// 	stream << input_int;
// 	return stream.str();
// }

// string int_to_6_string(int input_int){
// 	ostringstream stream;
//     // can not use this mothod, if not output will have other signal
//     // ss << setfill('0') << setw(6) << input_int << endl;  
//     stream << input_int;
//     string tmp = stream.str();
//     int zero_length = 6 - tmp.length(); 
//     for(int i=0; i<zero_length; i++){
//         tmp = "0" + tmp;
//     }
//     return tmp;
// }

bool dirExists(const string& dirname){
	// flag 0 means judge whether the file exists
	int flag = access(dirname.c_str(), 0);
	if(flag == 0){
		return true;
	}
	else{
		return false;
	}
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
	if(save_image){
		string jpeg_dir = save_root + "/JPEGImages/";
		string anno_dir = save_root + "/Annotations/";
		if (not(dirExists(jpeg_dir))){
			string cmd = "mkdir " + jpeg_dir;
			system(cmd.c_str());
		} 
		if (not(dirExists(anno_dir))){
			string cmd = "mkdir " + anno_dir;
			system(cmd.c_str()); 
		}
	}
	
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
			line(temp, Point(1, box.y+box.height), Point(1920, box.y+box.height), Scalar(0,255,0), 2, CV_AA);
			line(temp, Point(box.x+box.width, 1), Point(box.x+box.width, 1080), Scalar(0,255,0), 2, CV_AA);
			rectangle(temp, box, Scalar(0, 255, 0), 2, 1);
			imshow("Tracking", temp);//显示
		}
	}

	//Rect2d box(270, 120, 180, 260);
	tracker->init(frame, box);
	int num_frame = 1;
	int num_saved_pic = saved_pic_start;
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

		if (save_image)
		{
			if (num_frame % interval == 0){
				string six_string = int_to_6_string(num_saved_pic); 
				string pic_name = six_string + ".jpg";
				string xml_name = six_string + ".xml";
				string pic_dir = save_root + "/JPEGImages/" + pic_name;
				string xml_dir = save_root + "/Annotations/" + xml_name; 
				imwrite(pic_dir, frame);
    			if(saveXML(pic_name, pic_dir, track_object_type, box.x, box.y, 
					box.x+box.width, box.y+box.height, xml_dir) == FAILURE){
					return 1;
				}
				++ num_saved_pic;
			}
			++ num_frame;
			if(num_frame % 100 == 0){
				printf("have handled %d frames and have saved %d pictures\n", num_frame, num_saved_pic-saved_pic_start+1);
			}
		}

		if(show_image){
			rectangle(frame, box, Scalar(255, 0, 0), 2, 1);//如果存储图片就不画框
			imshow("Tracking", frame);
			int k=waitKey(1);
			if(k == 27)
				break;
			if(k == 32){
				gotBB = false;
				while(!gotBB){
					//只要不再次按下鼠标左键触发事件,则程序显示的一直是if条件里面被矩形函数处理过的temp图像，如果再次按下鼠标左键就进入if，不断更新被画矩形函数处理过的temp，因为处理速度快所以看起来画矩形的过程是连续的没有卡顿，因为每次重新画都是在没有框的基础上画出新的框因为人眼的残影延迟所以不会有拖影现象。每次更新矩形框的传入数据是重新被img（没有框）的数据覆盖的temp（即img.data==temp.data）和通过回调函数更新了的Box记录的坐标点数据。
					waitKey(1);//维持imshow
					if(drawing_box){//不断更新正在画的矩形
						frame.copyTo(temp);//这句放在这里是保证了每次更新矩形框都是在没有原图的基础上更新矩形框。
						line(temp, Point(1, box.y+box.height), Point(1920, box.y+box.height), Scalar(0,255,0), 2, CV_AA);
						line(temp, Point(box.x+box.width, 1), Point(box.x+box.width, 1080), Scalar(0,255,0), 2, CV_AA);
						rectangle(temp, box, Scalar(0, 255, 0), 2, 1);
						imshow("Tracking", temp);//显示
					}
				}
				tracker->clear();
				// There must be a intermediate quantity, otherwise it doesn't work
				Ptr<TrackerKCF> tracker2 = TrackerKCF::createTracker();
				tracker = tracker2;
				tracker -> init(frame, box);
			}
		}
	}
	printf("finished tagging this video, and save %d pictures and xmls\n", num_saved_pic-saved_pic_start+1);
	printf("now the last picture is %d\n", num_saved_pic-1);
}
