#pragma once
#ifdef _WIN32
#include <windows.h>
#endif

#include <fstream>
#include <iostream>
#include <numeric>
#include <filesystem>
#include <format>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "global.h"

using namespace std;
using namespace cv;
using namespace std::filesystem;
using namespace std::chrono;

class Logger : public DebugMessage {
    bool logEnable;
    bool debugMode;
    bool writeLive;             // for internal usage
    int numChannels, numPages;  // at most 9 channels for each page
    int targetLiveChannel;      // live channel to be drown
    bool updateIS;

    vector<tm> preTms;
    vector<Mat> canvases;

    vector<int> vchStatesPre;  /// vch states (0: non-connected, 1: connected)
    vector<int> maxFireProbs, maxSmokeProbs, accNumFires, accNumSmokes;
    vector<int> maxFireProbsDay, maxSmokeProbsDay, accNumFiresDay, accNumSmokesDay;

    vector<bool> writeChImgs;

    vector<int> preTimes;
    vector<bool> pageUpdated;
    vector<bool> vchUpdated;

    // for debug
    VideoWriter writer;

   private:
    int toABTime(string &str);
    void newDate(int numChannels, tm *inputTm);
    bool checkDirectories(int numChannels);
    string normLogPath(string path);  // get normalized path for windows and linux

    void drawCanvase(Mat &frame, int vchID, tm *curTm, int msec);

    void writeIS(Config &cfg, vector<ODRecord> &odRcds, vector<CCRecord> &ccRcds);
    void writeCntLine(ofstream &f, CntLine *c, CntLine *p = NULL);
    void writeZone(ofstream &f, Zone *c, Zone *p = NULL);

    void deleteNowLive(int sec, bool boostMode);
    void writeNowLive(int sec, int msec, bool boostMode);
    void writeNowOD(ODRecord &odRcd, int vchID, string dayInfo, int sec);
    void writeNowFD(int vchID, string dayInfo, int sec);
    void writeNowCC(CCRecord &ccRcd, int vchID, string dayInfo, int sec);

    string getNowCntLine(CntLine *c);
    string getNowZone(Zone *c);

   public:
    Logger(Config &cfg);
    ~Logger();

    void writeChInfo();

    bool createLog();
    bool needToDraw(int vchID);
    bool checkCmdOD(Config &cfg, vector<ODRecord> &odRcds, vector<FDRecord> &fdRcds, vector<CCRecord> &ccRcds);
    bool checkCmdCC(Config &cfg, vector<CCRecord> &ccRcds);
    void writeData(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd, Mat &frame, unsigned int &frameCnt,
                   int vchID, system_clock::time_point now);
    void destroy();

    // void getNowLive(vector<Mat>& nowMats);
    // void getNowOD(vector<string>& nowOD);
    // void getNowFD(vector<string>& nowFD);
    // void getNowCC(vector<string>& nowCC);
};