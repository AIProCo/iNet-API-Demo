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

using namespace std;
using namespace cv;

class VideoStreamer {
   public:
    Config *pCfg;
    vector<ODRecord> *pOdRcds;
    vector<CCRecord> *pCcRcds;

    int numChannels;
    vector<string> inputs;
    vector<string> outputs;

    vector<VideoWriter> videoWriters;
    vector<VideoCapture> captures;

    VideoStreamer(Config &cfg, std::vector<ODRecord> &odRcds, std::vector<CCRecord> &ccRcds);
    ~VideoStreamer();

    void destroy();  // explicit destroy function. (cuz destructor is called randomly)
    bool read(Mat &frame, int vchID);

   private:
    void init();

   public:
    VideoWriter &operator[](int idx) {
        return videoWriters[idx];
    }
    void write(Mat &frame, int vchID) {
        videoWriters[vchID] << frame;
    }
};
