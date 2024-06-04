/*==============================================================================
* Copyright 2024 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/// System
#define PERSON 0  /// person should be the first object in a mapping list

#define NUM_CLASSES_FD 2  /// number of fire detection classes(should be compatible with the FD model)
#define SMOKE 0
#define FIRE 1

#define NET_WIDTH_OD 960   /// net width for od
#define NET_HEIGHT_OD 544  /// net height for od

#define NET_WIDTH_FD 640   /// net width for fd
#define NET_HEIGHT_FD 360  /// net height for fd

#define NUM_CC_LEVELS 4
#define NET_WIDTH_CC 1920   /// net width for cc
#define NET_HEIGHT_CC 1080  /// net height for cc
#define OUT_WIDTH_CC 2048   /// net width for cc
#define OUT_HEIGHT_CC 1536  /// net height for cc

#define NUM_ATTRIBUTES 30  /// number of attributes(should be compatible with the PAR model)
#define ATT_GENDER 0       /// gender should be the first att in a mapping list
#define ATT_AGE_CHILD 1
#define ATT_AGE_ADULT 2
#define ATT_AGE_ELDER 3
#define ATT_HAIR_LEN_SHORT 4
#define ATT_HAIR_LEN_LONG 5
#define ATT_UBODY_LEN_SHORT 6
#define ATT_UBODY_COL_BLACK 7
#define ATT_UBODY_COL_BLUE 8
#define ATT_UBODY_COL_BROWN 9
#define ATT_UBODY_COL_GREEN 10
#define ATT_UBODY_COL_GRAY 11
#define ATT_UBODY_COL_PINK 12
#define ATT_UBODY_COL_PURPLE 13
#define ATT_UBODY_COL_RED 14
#define ATT_UBODY_COL_WHITE 15
#define ATT_UBODY_COL_YELLOW 16
#define ATT_UBODY_COL_OTHER 17
#define ATT_LBODY_LEN_SHORT 18
#define ATT_LBODY_COL_BLACK 19
#define ATT_LBODY_COL_BLUE 20
#define ATT_LBODY_COL_BROWN 21
#define ATT_LBODY_COL_GRAY 22
#define ATT_LBODY_COL_WHITE 23
#define ATT_LBODY_COL_OTHER 24
#define ATT_LBODY_TYPE_TROUSER_SHORT 25
#define ATT_LBODY_TYPE_SKIRT_DRESS 26
#define ATT_BACKPACK 27
#define ATT_BAG 28
#define ATT_HAT 29

#define NUM_FD_CLASSES 4
#define FD_CLASS_BOTH 0
#define FD_CLASS_FIRE 1
#define FD_CLASS_NONE 2
#define FD_CLASS_SMOKE 3

#define NUM_GENDERS 2
#define MALE 0
#define FEMALE 1

#define NUM_AGE_GROUPS 3
#define CHILD_GROUP 0
#define ADULT_GROUP 1
#define ELDER_GROUP 2

/// data structure
#define LOG_ENABLE true

#ifdef _WIN32
#define AIPRO_PATH "c:/aipro/"
#else
#define AIPRO_PATH "/home/aipro/jetson/"
#endif

#define ROOT_PATH AIPRO_PATH "data/"
#define CONFIG_PATH ROOT_PATH "config"
#define CHIMGS_PATH ROOT_PATH "chimgs"
#define CNT_PATH ROOT_PATH "cnt"
#define CC_PATH ROOT_PATH "cc"
#define FD_PATH ROOT_PATH "fd"
#define LOG_PATH ROOT_PATH "log"
#define VIDEO_OUT_PATH AIPRO_PATH "inet/videos"

/// s2e commands
#define CMD_INSERT_LINE 0
#define CMD_MODIFY_LINE 1
#define CMD_REMOVE_LINE 2
#define CMD_INSERT_ZONE 3
#define CMD_MODIFY_ZONE 4
#define CMD_REMOVE_ZONE 5
#define CMD_INSERT_CCZONE 6
#define CMD_MODIFY_CCZONE 7
#define CMD_REMOVE_CCZONE 8
#define CMD_UPDATE_SCORETHS 50
#define CMD_CLEARLOG 100
#define CMD_REMOVE_ALL_LINES_ZONES 101
#define CMD_REMOVE_ALL_LINES 102
#define CMD_REMOVE_ALL_ZONES 103
#define CMD_REMOVE_ALL_CCZONES 104
#define CMD_TERMINATE_PROGRAM 200

/// IS_MODE
#define IS_PEOPLE_COUNTING 0
#define IS_RESTRICTED_AREA 1

typedef unsigned char uchar;
typedef unsigned int uint;

struct Zone {
    bool enabled;  /// enable flag
    int zoneID;    /// Unique zone id
    int vchID;     /// vchID where the zone exists

    int isMode;    /// IS mode(0: people counting, 1: restricted area)
    int preTotal;  /// for internal usage in logger

    std::vector<cv::Point> pts;                  /// corner points (should be larger than 2)
    int curPeople[NUM_GENDERS][NUM_AGE_GROUPS];  /// current people in the zone
    int hitMap[NUM_GENDERS][NUM_AGE_GROUPS];     /// total people in the zone
};

struct CntLine {
    bool enabled;   /// enable flag
    int clineID;    /// unique counting line id
    int vchID;      /// vchID where the counting line exits
    int direction;  /// counting line direction (0:horizonal, 1:vertical)

    int isMode;    /// IS mode(0: people counting, 1: restricted area)
    int preTotal;  /// for internal usage in logger

    cv::Point pts[2];                          /// 2 end-points
    int totalUL[NUM_GENDERS][NUM_AGE_GROUPS];  // number of total object that move up or left
    int totalDR[NUM_GENDERS][NUM_AGE_GROUPS];  // number of total object that move down or right
};

struct CCZone {
    bool enabled;                       /// enable flag
    int ccZoneID;                       /// Unique czone id
    int vchID;                          /// vchID where the zone exists
    std::vector<cv::Point> pts;         /// corner points (should be larger than 2)
    int ccLevelThs[NUM_CC_LEVELS - 1];  /// ccLevel1, ccLevel2, ccLevel3
    int ccLevel;                        /// current ccLevel
    int preCCLevel;                     /// previous ccLevel
    cv::Mat mask;                       /// for internal usage in CrowdCounter(for GPU and CPU modes)

#ifdef _CPU_INFER
    cv::Mat canvas, roiCanvas;            /// for internal usage in CrowdCounter(only for CPU)
    cv::Point roiTL, roiBR, roiBRScaled;  /// for internal usage in CrowdCounter(only for CPU)
    cv::Size roiScaledSize;
    double sH, sW;
    // int roiScaledH, roiScaledW;

    void setCanvas(cv::Mat &frame) {
        if (canvas.empty())
            initCanvas(frame);

        cv::Mat roi = frame(cv::Rect(roiTL, roiBR));
        cv::Mat roiScaled;

        if (roi.size() == roiScaledSize)
            roiScaled = roi;
        else
            cv::resize(roi, roiScaled, roiScaledSize);

        roiCanvas = roiScaled.mul(mask);
    }

    void initCanvas(cv::Mat &frame) {
        sH = (float)NET_HEIGHT_CC / frame.rows;
        sW = (float)NET_WIDTH_CC / frame.cols;

        canvas = cv::Mat::zeros(NET_HEIGHT_CC, NET_WIDTH_CC, CV_8UC3);

        roiTL = cv::Point(INT_MAX, INT_MAX);
        roiBR = cv::Point(0, 0);

        for (auto &pt : pts) {
            if (pt.x < roiTL.x)
                roiTL.x = pt.x;

            if (pt.y < roiTL.y)
                roiTL.y = pt.y;

            if (pt.x > roiBR.x)
                roiBR.x = pt.x;

            if (pt.y > roiTL.y)
                roiBR.y = pt.y;
        }

        std::vector<cv::Point> movedPts;
        for (auto &pt : pts) {
            cv::Point movedPt;
            movedPt.x = (pt.x - roiTL.x) * sW;
            movedPt.y = (pt.y - roiTL.y) * sH;
            movedPts.push_back(movedPt);
        }

        roiScaledSize.height = (roiBR.y - roiTL.y) * sH;
        roiScaledSize.width = (roiBR.x - roiTL.x) * sW;

        roiCanvas = canvas(cv::Rect(0, 0, roiScaledSize.width, roiScaledSize.height));
        mask = cv::Mat::zeros(roiScaledSize.height, roiScaledSize.width, CV_8UC3);
        cv::fillConvexPoly(mask, movedPts, cv::Scalar(1, 1, 1));
    }
#endif
    int maxCC;                              /// for internal usage in logger
    int maxCCDay;                           /// for internal usage in logger
    int accCCLevels[NUM_CC_LEVELS - 1];     /// for internal usage in logger
    int accCCLevelsDay[NUM_CC_LEVELS - 1];  /// for internal usage in logger

    std::deque<int> ccNums;

    void pushCCNum(int ccNum) {
        if (ccNums.size() <= 0)
            return;

        if (ccNum > 10) {
            int preCCNum = ccNums.back();

            if (preCCNum > 10)
                ccNum = ccNum * 0.8f + preCCNum * 0.2f;
        }

        ccNums.pop_front();
        ccNums.push_back(ccNum);

        ccLevel = 0;
        for (int l = 2; l >= 0; l--) {
            if (ccNum > ccLevelThs[l]) {
                ccLevel = l + 1;
                break;
            }
        }
    }
};

struct ODRecord {
    std::vector<Zone> zones;        // zones
    std::vector<CntLine> cntLines;  // cntLine
};

struct FDRecord {
    std::vector<std::deque<float>> fireProbsMul;   // fire probability
    std::vector<std::deque<float>> smokeProbsMul;  // smoke probability
    std::vector<int> fireEvents;                   // fire event
    std::vector<int> smokeEvents;                  // smoke event
    std::vector<int> afterFireEvents;              // for internal usage in logger
};

struct CCRecord {
    std::vector<std::deque<int>> ccNumFrames;  // people in the whole frame
    std::vector<CCZone> ccZones;               // ccZones
};

/// data structure for pedestrian attributes (gender, age, has backpack, etc)
struct PedAtts {
    int setCnt;                  /// frame count after the last PAR inference
    float atts[NUM_ATTRIBUTES];  /// Attributes to be extracted

    static bool getGenderAtt(PedAtts &patts) {
        return (patts.atts[ATT_GENDER] > 0.5);
    }

    static void getGenderAtt(PedAtts &patts, bool &isFemale, int &prob) {
        isFemale = patts.atts[ATT_GENDER] > 0.5;
        prob =
            isFemale ? (int)(patts.atts[ATT_GENDER] * 100 + 0.5f) : (int)((1.0f - patts.atts[ATT_GENDER]) * 100 + 0.5f);
    }

    static int getAgeGroupAtt(PedAtts &patts) {
        int ageGroup;

        if (patts.atts[ATT_AGE_CHILD] > patts.atts[ATT_AGE_ADULT] &&
            patts.atts[ATT_AGE_CHILD] > patts.atts[ATT_AGE_ELDER]) {
            ageGroup = CHILD_GROUP;
        } else {
            if (patts.atts[ATT_AGE_ADULT] > patts.atts[ATT_AGE_ELDER]) {
                ageGroup = ADULT_GROUP;
            } else {
                ageGroup = ELDER_GROUP;
            }
        }

        return ageGroup;
    }

    static void getAgeGroupAtt(PedAtts &patts, int &ageGroup, int &prob) {
        if (patts.atts[ATT_AGE_CHILD] > patts.atts[ATT_AGE_ADULT] &&
            patts.atts[ATT_AGE_CHILD] > patts.atts[ATT_AGE_ELDER]) {
            ageGroup = CHILD_GROUP;
            prob = patts.atts[ATT_AGE_CHILD] * 100 + 0.5f;
        } else {
            if (patts.atts[ATT_AGE_ADULT] > patts.atts[ATT_AGE_ELDER]) {
                ageGroup = ADULT_GROUP;
                prob = patts.atts[ATT_AGE_ADULT] * 100 + 0.5f;
            } else {
                ageGroup = ELDER_GROUP;
                prob = patts.atts[ATT_AGE_ELDER] * 100 + 0.5f;
            }
        }
    }
};

/// data structure for object detection result
struct DetBox {
    int x, y, w, h;   /// (x, y): top-left corner, (w, h): width & height of bounded box
    int rx, ry;       /// (rx, ry): reference position for counting
    int objID;        /// class of object - from range [0, classes-1]
    uint trackID;     /// tracking id (0: untracked, 1 - inf : tracked object)
    int vchID;        /// video channel id
    uint frameCnt;    /// frame cnt
    float prob;       /// confidence - probability that the object was found correctly
    time_t inTime;    /// time when this object is detected (this should be set in Counter)
    bool onBoundary;  /// located on frame boundary

    int rxP, ryP;       /// for internal usage: reference position in the previous frame for counting
    uint lastFrameCnt;  /// for internal usage: frameCnt of the last counting

    float distVar;          /// box center variation after temporal pooling
    uchar justCountedLine;  /// for emphasizing the object just counted (15:lastest - 0:no action)
    uchar justCountedZone;  /// for emphasizing the object just counted (15:lastest - 0:no action)

    PedAtts patts;  /// PAR info
};

class Logger;

struct Config {
    std::string key;                       /// authorization Key
    uint frameLimit;                       /// number of frames to be processed
    std::vector<std::string> inputFiles;   /// list of the input files
    std::vector<std::string> outputFiles;  /// list of the output files
    bool recording;                        /// record output videos
    bool debugMode;                        /// output debug info and frames
    bool boostMode;                        /// enable boost mode(minimize delay)
    bool igpuEnable;                       /// use igpu if present
    bool logEnable;

    // config
    int maxBufferSize;              /// maximun input buffer size
    int numChannels;                /// number of video channels (unlimited)
    std::vector<int> vchStates;     /// vch states (0: non-connected, 1: connected)
    std::vector<int> frameWidths;   /// widths of the input frames
    std::vector<int> frameHeights;  /// heights of the input frames
    std::vector<float> fpss;        /// fpss of the input frames

    // od config
    bool odEnable;            /// Enable object detection and tracking
    std::string odModelFile;  /// path to the od config file (ex:aipro_b5.trt)
    int odNetWidth;           /// width of the od model input
    int odNetHeight;          /// height of the od model input
    std::vector<float> odScaleFactors;
    std::vector<float> odScaleFactorsInv;
    float odScoreTh;  /// threshold for filtering low confident detections
    int odBatchSize;  /// batch size of the od model
    std::vector<std::string> odIDMapping;
    int numClasses;                /// number of classes
    std::vector<bool> odChannels;  /// flags for indicating object detection channels
    std::vector<bool> fdChannels;  /// flags for indicating fire detection channels
    std::vector<bool> ccChannels;  /// flags for indicating crowd counting channels

    // fd config
    bool fdEnable;            /// Enable fire detection and tracking
    std::string fdModelFile;  /// path to the od config file (ex:aipro_b5.trt)
    int fdNetWidth;           /// width of the fd model input
    int fdNetHeight;          /// height of the fd model input
    std::vector<float> fdScaleFactors;
    std::vector<float> fdScaleFactorsInv;
    float fdScoreTh;   /// threshold for filtering low confident detections
    int fdBatchSize;   /// batch size of the fd model
    int fdWindowSize;  /// window size for fire and smoke detection history
    std::vector<std::string> fdIDMapping;
    int fdNumClasses;  /// number of classes
    int fdPeriod;      /// fire detection period

    // tracking
    int longLastingObjTh;  /// threshold for checking long-lasting objects in second
    float noMoveTh;        /// threshold for checking no movement objects

    // counting
    int debouncingTh;         /// debounce counting results over successive frames (suppress duplicated counting)
    int lineEmphasizePeroid;  /// line emphasize peroid

    // par config
    bool parEnable;            /// enable par
    bool parLightMode;         /// enable the light mode
    std::string parModelFile;  /// par model file (ex:res50_256_128_a1_b32.onnx)
    std::vector<std::string> parIDMapping;
    int numAtts;          /// number of attributes (should be matched to par model)
    int attUpdatePeriod;  /// attribute update period
    int parBatchSize;     /// batch size of the PAR onnx model

    // crowd counting
    bool ccEnable;    /// enable crowd counting
    int ccNetWidth;   /// width of the crowd counting model input
    int ccNetHeight;  /// height of the crowd counting model input
    std::vector<float> ccScaleFactors;
    std::vector<float> ccScaleFactorsInv;
    std::string ccModelFile;  /// crowd counting model file
    int ccWindowSize;
    int ccPeriod;  /// fire detection period

    std::function<void(std::string)> lg;
};