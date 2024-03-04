#pragma once
#include <Windows.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/videoio.hpp>

#include "global.h"
#include "cmat.h"

using namespace std;
using namespace cv;

class CameraStreamer {
   public:
    Config *pCfg;
    ODRecord *pOdRcd;
    FDRecord *pFdRcd;
    CCRecord *pCcRcd;

    int numChannels;
    int maxBufferSize;
    vector<unsigned int> frameCnts;  // fpss are casted from double to int

    vector<string> inputs;
    vector<string> outputs;

    bool stopFlag;
    int initSleepPeriod;
    int sleepPeriod;

    CMats cmats;
    vector<VideoWriter> videoWriters;

    std::function<void(std::string)> lg;  // for writing log

    std::vector<std::thread *> cameraThreads;  // thread that run camera capture process
    CameraStreamer(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);
    void destory();  // explicit destory function. (cuz destructor is called randomly)

   private:
    /// <summary>
    /// If ip camera is disconnected, try to reconnect in 5sec.<para/>
    /// Trial to reconnect fail -> wait for 30 sec to try again
    /// </summary>
    /// <param name="index">: ipcamera index</param>
    /// <param name="input_q">: concurrent queue store original frames from ip cam</param>
    void keepConnected(int vchID);

    /// <summary>
    /// Grab frame from ipcam stream
    /// </summary>
    /// <param name="capture">: Videocapture pointer</param>
    /// <param name="index">: ipcamera index</param>
    /// <param name="input_q">: concurrent queue store original frames from ip cam</param>
    /// <returns>boolean result of grab frame</returns>
    bool working(cv::VideoCapture *capture, int vchID);  // grab frame from ipcam stream

   public:
    bool empty() {
        return cmats.empty();
    }
    bool try_pop(CMat &cmat) {
        return cmats.try_pop(cmat);
    }
    int unsafe_size() {
        return cmats.unsafe_size();
    }
    VideoWriter &operator[](int idx) {
        return videoWriters[idx];
    }
};
