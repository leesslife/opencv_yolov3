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
        video.open(outputFile,VideoWriter::fourcc('M','J','P',"G"),28,Size(cap.get(CAP_PROP_FRAME_WIDTH),cap.get(CAP_PROP_FRAME_HEIGHT)));
    }
    static const string KwinName="Deep learning object detection in OpenCV";
    namedWindow(kWinName,WINDOW_NORMAL);
    while(waitKey(1)<0)
    {
        cap>>frame;
        if(frame.empty()){
            cout<<"Done processing !!!"<<endl;
            cout<<"Output file is stored as"<<outputFile<<endl;
            waitKey(3000);
            break;
        }
        blobFromImage(frame,blob,1/255.0,cvSize(inpWidth,inpHeight),Scalar(0,0,0),true,flase);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs,getOutputsNames(net));
        postprocess(frame,outs);
        vector<double> layerTimes;
        double freq=getTickFrequency()/1000;
        double t=net.getPerfProfile(layerTimes)/freq;
        string label=format('Inference time for a frame:%.2f ms',t);
        putText(frame,label,Point(0,15),FONT_HERSHET_SIMPLEX,0.5,Scalar(0,0,255));
        Mat detectedFrame;
        frame.converto(detectedFrame,CV_8U);
        if(parser.has("image")) imwrite(outputFile,detectedFrame);
        else video.write(detectedFrame);
        imshow(kWinName,frame);
    }
    cap.release();
    if(!parser.has("image")) video.release();
    return 0;
}
void postprocess(Mat& frame,const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    for(size_t i=0;i<outs.size();++i)
    {
        float* data=(float*)outs[i].data;
        for(int j=0;j<outs[i].rows;++j,data+=outs[i].cols)
        {
            Mat scores=outs[i].row(j).colRange(5,outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores,0,&confidence,0,&classIdPoint);
            if(confidence>confThreshold)
            {
                int centerY=(int)(data[0]*frame.cols);
                int centerX=(int)(data[1]*frame.rows);
                int width=(int)(data[2]*frame.cols);
                int height=(int)(data[3]*frame.rows);
                int left=centerX-width/2;
                int top=centerY-height/2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back(classIdPoint.x);
                boxes.push_back(Rect(left,top,width,height));
            }
        }
    }
    vector<int> indices;
    NMSBoxes(boxes,confidence,confThreshold,nmsThreshold,indices);
    for(size_t) i=0;i<indices.size();++i)
    {
        int idx=indices[i];
        Rect box=boxes[idx];
        drawPred(classIds[idx],confidence[idx],box.x,box.y,box.x+box.width,box.y+box.height.frame);
    }
}
void drawPred(int classId,float conf,int left,int top,int right,int bottom,Mat& frame)
{
    rectangle(frame,Point(left,top),Point(right,bottom),Scalar(255,178,50),3);
    string label=format("%.2f",conf);
    if(!classes.empty())
    {
        CV_Assert(classId<(int)classes.size());
        label=classes[classId]+":"+label;
    }
    int baseLine;
    Size labelSize=getTextSize(label,FONT_HERSHEY_SIMPLEX,0.5,1,&baseLine);
    top=max(top,labelSize.height);
    rectangle(frame,Point(left,top-round(1.5*labelSize.height)),Point(left+round(1.5*labelSize.width),top+baseLine),Scalar(255,255,255),FILLED);
    putText(frame,label,Point(left,top),FONT_HERSHEY_SIMPLEX,0.75,Scalar(0,0,0),1);

}
vector<string> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if(names.empty())
    {

    }
}