#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
const char* keys=
"{help h usage ? | |Usage examples:\n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image  }"
"{video v        |<none>| input video  }";
using namespace cv;
using namespace dnn;
using namespace std;

float confThreshold=0.5;
float nmsThreshold=0.4;
int inpWidth=416;
int inpHeight=416;
vect<string> classes;
void postprocess(Mat& frame,const vector<Mat>& out);
void drawPred(int classId,float conf,int left,int top,int right,int bottom,Mat& frame);
vector<string> getOutputsNames(const Net& net);

int main(int argc,char** argv)
{
    CommandLineParser parser(argc,argv,keys);
    parser.about("Use this script to run object detection using YOLO3 in Opencv");
    if(parser.has("help")){
        parser.printMessage();
        return 0;
    }
    string classesFile="coco.names";
    ifstream ifs(classesFIle.c_str());
    string line;
    while(getline(ifs,line)) classes.push_back(line);
    string modelConfiguration="yolov3.cfg";
    string modelWeights="yolov3.weights";
    Net net=readNetFromDarknet(modelConfiguration,modelWeights);
    net.setPreferableBackend(modelConfiguration,modelWeights);
    net.setPreferableTarget(DNN_TARGET_CPU);
    string str,outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat Frame,blob;
    try{
        outputFile="yolo_out_cpp.avi";
        if(parser.has("image"))
        {
            str=parser.get<string>("image");
            ifstream ifile(str);
            if(!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4,str.end(),"_yolo_out_cpp.jpg");
            outputFile=str;
        }
        else if(parser.has("video"))
        {
            str=parser.get<string>("video");
            ifstream ifile(str);
            cap.open(str);
            str.replace(str.end()-4,str.end(),"_yolo_out_cpp.avi");
            outputFile=str;
        }
        else cap.open(parser.get<int>("device"));
    }
    catch(...){
        cout<<"could not open the input image/video stream"<<endl;
        return 0;
    }
    if(!parser.has("image")){
        video.open(outputFile,VideoWriter::fourcc('M','J','P',"G"),28,Size(cap.get(CAP_PROP_FRAME_WIDTH,cap.get(CAP_PROP_FRAME_HEIGHT)));
    }
}