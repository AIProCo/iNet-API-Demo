/*==============================================================================
* Copyright 2022 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

/// System
#define PERSON 0  /// person should be the first object in a mapping list

#define NUM_ATTRIBUTES 4  /// number of attributes(should be compatible with the PAR model)
#define ATT_GENDER 0      /// gender should be the first att in a mapping list
#define ATT_CHILD 1
#define ATT_ADULT 2
#define ATT_ELDER 3

#define NUM_GENDERS 2
#define MALE 0
#define FEMALE 1

#define NUM_AGE_GROUPS 3
#define CHILD_GROUP 0
#define ADULT_GROUP 1
#define ELDER_GROUP 2

#define NUM_SKEL_KEYPOINTS 17

typedef unsigned char uchar;
typedef unsigned int uint;

struct Zone {
    int zoneID;                  /// Unique zone id
    int vchID;                   /// vchID where the zone exists
    bool isRestricted;           /// restricted zone
    std::vector<cv::Point> pts;  /// corner points (should be larger than 2)

    int curPeople[NUM_GENDERS][NUM_AGE_GROUPS];  /// current people in the zone
    int hitMap[NUM_GENDERS][NUM_AGE_GROUPS];     /// total people in the zone
};

struct CntLine {
    int clineID;             /// unique counting line id
    int vchID;               /// vchID where the counting line exits
    int direction;           /// counting line direction (0:horizonal, 1:vertical)
    cv::Point pts[2];        /// 2 end-points

    int totalUL[NUM_GENDERS][NUM_AGE_GROUPS];  // number of total object that move up or left
    int totalDR[NUM_GENDERS][NUM_AGE_GROUPS];  // number of total object that move down or right
};

struct Record {
    std::vector<Zone> zones;        // zones
    std::vector<CntLine> cntLines;  // cntLine
};

/// data structure for pedestrian attributes (gender, age, has backpack, etc)
struct PedAtts {
    int setCnt;                  /// frame count after the last PAR inference
    float atts[NUM_ATTRIBUTES];  /// Attributes to be extracted

    static bool getGenderAtt(PedAtts& patts) {
        return (patts.atts[ATT_GENDER] > 0.5);
    }

    static void getGenderAtt(PedAtts& patts, bool& isFemale, int& prob) {
        isFemale = patts.atts[ATT_GENDER] > 0.5;
        prob =
            isFemale ? (int)(patts.atts[ATT_GENDER] * 100 + 0.5f) : (int)((1.0f - patts.atts[ATT_GENDER]) * 100 + 0.5f);
    }

    static int getAgeGroupAtt(PedAtts& patts) {
        int ageGroup;

        if (patts.atts[ATT_CHILD] > patts.atts[ATT_ADULT] && patts.atts[ATT_CHILD] > patts.atts[ATT_ELDER]) {
            ageGroup = CHILD_GROUP;
        }
        else {
            if (patts.atts[ATT_ADULT] > patts.atts[ATT_ELDER]) 
                ageGroup = ADULT_GROUP;
            else 
                ageGroup = ELDER_GROUP;
        }

        return ageGroup;
    }

    static void getAgeGroupAtt(PedAtts& patts, int& ageGroup, int& prob) {
        if (patts.atts[ATT_CHILD] > patts.atts[ATT_ADULT] && patts.atts[ATT_CHILD] > patts.atts[ATT_ELDER]) {
            ageGroup = CHILD_GROUP;
            prob = patts.atts[ATT_CHILD] * 100 + 0.5f;
        }
        else {
            if (patts.atts[ATT_ADULT] > patts.atts[ATT_ELDER]) {
                ageGroup = ADULT_GROUP;
                prob = patts.atts[ATT_ADULT] * 100 + 0.5f;
            }
            else {
                ageGroup = ELDER_GROUP;
                prob = patts.atts[ATT_ELDER] * 100 + 0.5f;
            }
        }
    }
};

/// Pose Information
const std::vector<std::pair<int, int>> cocoSkeletons = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
    {8, 10},  {9, 11},  {2, 3},   {1, 2},   {1, 3},   {2, 4},  {3, 5},  {4, 6}, {5, 7} };

/// class representing a single Keypoint in pose estimation
class SKeyPoint {
public:
    int pointIdx;
    float x;
    float y;
    float confScore;  /// confidence score
    SKeyPoint(int _pointIdx = 0, float _x = -1.0f, float _y = -1.0f, float _conf = 0.0f) {
        pointIdx = _pointIdx;
        x = _x;
        y = _y;
        confScore = _conf;
    }
    friend std::ostream& operator<<(std::ostream& os, const SKeyPoint& kpt) {
        os << "[" << kpt.pointIdx << "-(" << kpt.x << "," << kpt.y << ")," << kpt.confScore << "]";
        return os;
    }
};

/// data structure for a pose estimation result for a single person-type bbox
typedef std::vector<SKeyPoint> Skeleton;

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
    uint lastFrameCnt;  /// for internal usage
    float distVar;      /// box center variation after temporal pooling

    PedAtts patts;  /// PAR info

    Skeleton skel;      /// skel info
    float activity[4];  /// activity of the current skel compared to the previous skel (head, shoulders, hands, foots)
    int numFramesOT;    /// number of the stored activity values in the current clip

    int actID;      /// action ID in actIDMapping
    float actConf;  /// confidence of the current act
    int actSetCnt;  /// time elapsed since the last act info was detected
};

struct Config {
    std::string key;                       /// authorization Key
    uint frameLimit;                       /// number of frames to be processed
    int gpuID;                             /// GPU ID used for inference
    std::vector<std::string> inputFiles;   /// list of the input files
    std::vector<std::string> outputFiles;  /// list of the output files

    // od config
    bool odEnable;            /// Enable object detection and tracking
    std::string odModelFile;  /// path to the od config file (ex:aipro_b5.trt)
    int netWidth;             /// width of model input
    int netHeight;            /// height of model input
    float odScoreTh;          /// threshold for filtering low confident detections
    int odBatchSize;          /// batch size of the od model
    std::vector<std::string> odIDMapping;
    int numClasses;                   /// number of classes
    std::vector<bool> isMainChannel;  // flags for indicating main channels

    // tracking
    int frameStory;        /// the number of False-Negative detections, during which track_id will be kept
    int maxDist;           /// max distance in pixels between previous and current detection, to keep the same track_id
    int longLastingObjTh;  /// threshold for checking long-lasting objects in second
    float noMoveTh;        /// threshold for checking no movement objects

    // counting
    int debouncingTh;  /// debounce counting results over successive frames (suppress duplicated counting)

    // par config
    bool parEnable;            /// enable par
    std::string parModelFile;  /// par model file (ex:res50_256_128_a1_b32.onnx)
    std::vector<std::string> parIDMapping;
    int numAtts;          /// number of attributes (should be matched to par model)
    int attUpdatePeriod;  /// attribute update period
    int parBatchSize;     /// batch size of the PAR onnx model

    int numChannels;                /// number of video channels (unlimited)
    std::vector<int> frameWidths;   /// widths of the input frames
    std::vector<int> frameHeights;  /// heights of the input frames
    std::vector<float> fpss;        /// fpss of the input frames

    // pose config
    bool poseEnable;            /// Enable pose
    std::string poseModelFile;  /// pose model file (ex: hrnet_crowdHuman.onnx)
    float poseScoreTh;          /// threshold for filtering low score keypoinits
    int poseBatchSize;          /// batch size of the pose onnx model

    // action config
    bool actEnable;            /// enable action recognition
    std::string actModelFile;  /// act model file (ex: aipro_act_t17_b2.onnx)
    std::vector<std::string> actIDMapping;
    float actScoreTh;      /// threshold for filtering low confident actions
    float heatmapScoreTh;  /// threshold for filtering low confident keypoint in heatmap generation
    int actBatchSize;      /// batch size of the act onnx model
    int actUpdatePeriod;   /// action info updata period(= act model inference period)
    int actLastPeriod;     /// time period to keep the action info (used in tracking)
    bool multiPersons;     /// generate clip using multiple persons
    int clipLength;        /// length of the clip
    int missingLimit;      /// frame-missing-tolerance limit
    int maxNumClips;       /// maximum number of clips to be stored at a time

    Record rcd;
};

const std::vector<std::string> aipro_t17 = {
    "hand on mouth",   "pick up",   "throw", "sit down",    "stand up",   "clapping",
    "reading/writing", "hand wave", "kick",  "cross hands", "staggering", "fall down",
    "punch/slap",      "push",      "walk",  "squat down",  "run" };