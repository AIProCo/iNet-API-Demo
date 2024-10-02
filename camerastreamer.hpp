#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif

#include <iostream>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <vector>

#include "global.h"
#include "util.h"

#include "logger.h"

using namespace std;
using namespace cv;

class CameraStreamer : public DebugMessage {
   public:
    ODRecord *pOdRcd;
    FDRecord *pFdRcd;
    CCRecord *pCcRcd;
    Logger *pLogger;

    int numChannels;
    int maxBufferSize;
    vector<unsigned int> frameCnts;  // fpss are casted from double to int

    vector<string> inputs;
    vector<string> outputs;

    volatile bool stopFlag;
    int initSleepPeriod;
    vector<int> sleepPeriods;

    vector<CMats> cmatsAll;
    vector<VideoWriter> videoWriters;

    std::vector<std::thread *> cameraThreads;  // thread that run camera capture process
    CameraStreamer(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd, Logger *_pLogger);
    ~CameraStreamer();

    void destroy();  // explicit destroy function. (cuz destructor is called randomly)

   private:
    /*void lg(std::string msg) {
        pCfg->lg(msg);
    }*/

    /// <summary>
    /// If ip camera is disconnected, try to reconnect in 5sec.<para/>
    /// Trial to reconnect fail -> wait for 30 sec to try again
    /// </summary>
    /// <param name="index">: ipcamera index</param>
    /// <param name="input_q">: concurrent queue store original frames from ip
    /// cam</param>
    void keepConnected(int vchID);

    /// <summary>
    /// Grab frame from ipcam stream
    /// </summary>
    /// <param name="capture">: Videocapture pointer</param>
    /// <param name="index">: ipcamera index</param>
    /// <param name="input_q">: concurrent queue store original frames from ip
    /// cam</param> <returns>boolean result of grab frame</returns>
    bool working(cv::VideoCapture *capture, int vchID);  // grab frame from ipcam stream

   public:
    bool empty(int vchID) {
        return cmatsAll[vchID].empty();
    }
    bool tryPop(CMat &cmat, int vchID) {
        return cmatsAll[vchID].try_pop(cmat);
    }
    int unsafeSize(int vchID) {
        return cmatsAll[vchID].unsafe_size();
    }
    int unsafeSizeMax() {
        int maxSize = 0;
        for (auto &cmats : cmatsAll)
            maxSize = max(maxSize, (int)cmats.unsafe_size());

        return maxSize;
    }
    VideoWriter &operator[](int idx) {
        return videoWriters[idx];
    }
    void write(Mat &frame, int vchID) {
        videoWriters[vchID] << frame;
    }
    int getPeriod(int vchID) {
        return sleepPeriods[vchID];
    }
};
