#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

using namespace std;
using namespace cv;


Mat src,img,ROI;
vector <Rect> cropRectList;
Rect cropRect(0,0,0,0);
Point P1(0,0);
Point P2(0,0);

bool clicked=false;
int i=0;

void checkBoundary(){
       //check croping rectangle exceed image boundary
       if(cropRect.width>img.cols-cropRect.x)
         cropRect.width=img.cols-cropRect.x;

       if(cropRect.height>img.rows-cropRect.y)
         cropRect.height=img.rows-cropRect.y;

        if(cropRect.x<0)
         cropRect.x=0;

       if(cropRect.y<0)
         cropRect.height=0;
}

void showImage(){
    img=src.clone();
    checkBoundary();
    if(cropRect.width>0&&cropRect.height>0){
        ROI=src(cropRect);
        imshow("cropped",ROI);
    }

    rectangle(img, cropRect, Scalar(0,255,0), 2, 8, 0 );
    imshow("bbimage",img);
}


class Frame{
public:
	vector<cv::Rect> boundingBoxes; //ground truth bounding boxes
	void appendBoundingBox(cv::Rect rect){
		boundingBoxes.push_back(rect);
	}
};

vector<Frame> loadGroundTruth(){
	ifstream in("groundtruth.txt");
	if (!in){
		cout << "Can't open groundtruth.txt\n";
		exit(0);
	}
	int x,y,w,h, frameNumber;
	char comma; //dummy
	vector<Frame> frames; // result
	while (
			(in >> frameNumber) && (in >> comma) &&
			(in >> x) && (in >> comma) &&
			  (in >> y) && (in >> comma) &&
			  (in >> w) && (in >> comma) &&
			  (in >> h)) {
		
			while (frames.size() < frameNumber)
				frames.push_back(Frame());
		
			if (frames.size() == frameNumber)
				frames.push_back(Frame());
			frames[frameNumber].appendBoundingBox(cv::Rect(x,y,w,h));
	}
	in.close();
	return frames;
};


void onMouse( int event, int x, int y, int f, void* ){
    switch(event){
        case  CV_EVENT_LBUTTONDOWN  :
                                        clicked=true;
                                        P1.x=x;
                                        P1.y=y;
                                        P2.x=x;
                                        P2.y=y;
                                        break;

        case  CV_EVENT_LBUTTONUP    :
                                        P2.x=x;
                                        P2.y=y;
                                        clicked=false;
                                        break;

        case  CV_EVENT_MOUSEMOVE    :
                                        if(clicked){
                                        P2.x=x;
                                        P2.y=y;
                                        }
                                        break;

        default                     :   break;


    }


    if(clicked){
     if(P1.x>P2.x){ cropRect.x=P2.x;
                       cropRect.width=P1.x-P2.x; }
        else {         cropRect.x=P1.x;
                       cropRect.width=P2.x-P1.x; }

        if(P1.y>P2.y){ cropRect.y=P2.y;
                       cropRect.height=P1.y-P2.y; }
        else {         cropRect.y=P1.y;
                       cropRect.height=P2.y-P1.y; }

    }


    showImage();
}

template <typename T>
std::string to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}

void drawRect(cv::Mat frame, cv::Rect rect, cv::Scalar color){

	cv::rectangle(
			frame,
			cv::Point(rect.x, rect.y),
			cv::Point(rect.x + rect.width, rect.y + rect.height),
			color, 2, 8, 0);

}

int main(int argc, char* argv[])
{

	if (argc < 2){
		cout << "Usage: ./video-annotation <VIDEO_FILE>\n";
		exit(0);
	}
			

    cout<<"Click and drag for create bounding box"<<endl<<endl;

    cout<<"------> Press 's' to save"<<endl<<endl;

    cout<<"------> Press '8' to move up"<<endl;
    cout<<"------> Press '2' to move down"<<endl;
    cout<<"------> Press '6' to move right"<<endl;
    cout<<"------> Press '4' to move left"<<endl<<endl;

    cout<<"------> Press 'w' increas top"<<endl;
    cout<<"------> Press 'x' increas bottom"<<endl;
    cout<<"------> Press 'd' increas right"<<endl;
    cout<<"------> Press 'a' increas left"<<endl<<endl;

    cout<<"------> Press 't' decrease top"<<endl;
    cout<<"------> Press 'b' decrease bottom"<<endl;
    cout<<"------> Press 'h' decrease right"<<endl;
    cout<<"------> Press 'f' decrease left"<<endl<<endl;

    cout<<"------> Press 'r' to reset"<<endl;
    cout<<"------> Press 'Esc' to quit"<<endl<<endl;

    vector <Frame> frames = loadGroundTruth();
  

    VideoCapture cap(argv[1]); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
	   return -1;
	
	ofstream myfile("groundtruth.txt", ios::out | ios::app);
	 
	
	int frameNumber = 0; 
	for(;;)
	{
	   namedWindow("bbimage",WINDOW_NORMAL);
	   setMouseCallback("bbimage",onMouse,NULL );
	   cap >> src;
	   
	   if(!src.empty()){
		   string strFrameNumber = "Frame number: " + to_string(frameNumber);
		   putText(src,  strFrameNumber, cv::Point(40,40), CV_FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0,200,0), 1, CV_AA);
		   imshow("bbimage",src);
	   }
	   
	   if (frameNumber < frames.size()){
		   for (int i = 0; i < frames[frameNumber].boundingBoxes.size(); i++){	
				//draw ground truth boundingbox
				drawRect(src, frames[frameNumber].boundingBoxes[i], cv::Scalar(255,0,0));
		   }
	   }
	   
	   vector<Rect> bboxes;
	   while(1){
		   char c=waitKey();
		   if(c=='s'&&ROI.data){
			   bboxes.push_back(cropRect);
			   cout<<"  Saved "<< endl;
		   }
	   
		   if(c=='6') cropRect.x++;
		   if(c=='4') cropRect.x--;
		   if(c=='8') cropRect.y--;
		   if(c=='2') cropRect.y++;
	   
		   if(c=='w') { cropRect.y--; cropRect.height++;}
		   if(c=='d') cropRect.width++;
		   if(c=='x') cropRect.height++;
		   if(c=='a') { cropRect.x--; cropRect.width++;}
	   
		   if(c=='t') { cropRect.y++; cropRect.height--;}
		   if(c=='h') cropRect.width--;
		   if(c=='b') cropRect.height--;
		   if(c=='f') { cropRect.x++; cropRect.width--;}
	   
		   if(c==27) break;
		   if(c=='n'){
			   for (int i = 0; i < bboxes.size(); i++){
				   myfile << 
						   frameNumber << "," << 
						   bboxes[i].x << "," << 
						   bboxes[i].y << "," << 
						   bboxes[i].width << "," <<
						   bboxes[i].height << endl;
			   }
			   break;
		   }
		   if(c=='r') {cropRect.x=0;cropRect.y=0;cropRect.width=0;cropRect.height=0;}
		   showImage();
	   }
	   frameNumber++;
	}

    return 0;
}
