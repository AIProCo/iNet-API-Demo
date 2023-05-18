/*==============================================================================
* Copyright 2023 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
#include <fstream>
#include <iostream>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

#include <opencv2/core/core.hpp>

// core
#include "global.h"
#include "generator.h"

// util
#include "json.hpp"
#include "util.h"

#define DRAW_DETECTION true
#define DRAW_FIRE_DETECTION true
#define DRAW_FIRE_DETECTION_COUNTING true
#define DRAW_POSE true
#define DRAW_ACTION true
#define DRAW_CNTLINE true
#define DRAW_CNTLINE_COUNTING true
#define DRAW_ZONE true
#define DRAW_ZONE_COUNTING true

#define CFG_FILEPATH "inputs/config.json"

#define OD_MD_FILEPATH "inputs/aipro_od_1_3.net"
#define OD_MD_CPU_FILEPATH "inputs/aipro_od_1_3_s.nez"

#define FD_MD_FILEPATH "inputs/aipro_fd_1_3.net"
#define FD_MD_CPU_FILEPATH "inputs/aipro_fd_1_3.nez"

#define PAR_MD_FILEPATH "inputs/aipro_par_1_3.net"
#define PAR_MD_CPU_FILEPATH "inputs/aipro_par_1_3.nez"

#define POSE_MD_FILEPATH "inputs/aipro_pose_1_3.net"
#define ACT_MD_FILEPATH "inputs/aipro_act_1_3.net"

using namespace std;
using namespace cv;
using json = nlohmann::json;

void printRecord(ODRecord &odRcd, unsigned int frameCnt);  // just for printing (can be omitted)
bool parseConfigAPI(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, VideoDir &videoDir);

void drawZones(ODRecord &odRcd, Mat &img, int vchID, double alpha);
void drawBoxes(Config &cfg, ODRecord &odRcd, Mat &img, vector<DetBox> &dboxes, int vchID, double alpha = 0.7,
               const vector<pair<int, int>> &skelPairs = cocoSkeletons);
void drawBoxesFD(Config &cfg, FDRecord &fdRcd, Mat &img, vector<FireBox> &fboxes, int vchID);

int main() {
    Config cfg;
    ODRecord odRcd;
    FDRecord fdRcd;
    VideoDir videoDir;

    if (!parseConfigAPI(cfg, odRcd, fdRcd, videoDir)) {
        cout << "Parsing Error!\n";
        return -1;
    }

    if (!initModel(cfg, odRcd, fdRcd)) {
        cout << "Initialization of the solution failed!\n";
        return -1;
    }

    bool endOfFrames = false;
    unsigned int frameCnt = 0;                 // should be unsigned int
    unsigned int frameLimit = cfg.frameLimit;  // number of frames to be processed
    int odBatchSize = cfg.odBatchSize;         // batch size of object detection

    vector<Mat> frames;
    vector<int> vchIDs;
    vector<unsigned int> frameCnts;
    clock_t start, middle, end;
    vector<float> infs0, infs1;

    while (1) {
        frameCnt++;

        for (int i = 0; i < cfg.numChannels; i++) {
            Mat frame;

            (videoDir[i]) >> frame;

            // If the frame is empty, break immediately
            if (frame.empty()) {
                cout << "End of Frames\n";
                endOfFrames = true;
                break;
            }

            frames.push_back(frame);
            vchIDs.push_back(i);
            frameCnts.push_back(frameCnt);

            // accumulate frames as many as batch
            if (frames.size() < odBatchSize)
                continue;

            vector<vector<DetBox>> dboxesMul(odBatchSize);

            start = clock();

            // batch inference for all models
            if (!runModel(dboxesMul, frames, vchIDs, frameCnts, cfg.odScoreTh, cfg.actScoreTh))
                break;

            middle = clock();

            vector<vector<FireBox>> fboxesMul(odBatchSize);

            // batch inference for all models
            if (!runModelFD(fboxesMul, frames, vchIDs, frameCnts, cfg.fdScoreTh))
                break;

            end = clock();

            for (int b = 0; b < odBatchSize; b++) {  // odBatchSize should be used instead of fdBatchSize
                drawBoxes(cfg, odRcd, frames[b], dboxesMul[b], vchIDs[b]);
                drawBoxesFD(cfg, fdRcd, frames[b], fboxesMul[b], vchIDs[b]);
                (videoDir[vchIDs[b]]) << frames[b];  // write a frame
            }

            float inf0 = (middle - start) / odBatchSize;
            float inf1 = (end - middle) / odBatchSize;

            if (frameCnt > 10 && frameCnt < 500) {  // skip the start frames and limit the number of elements
                infs0.push_back(inf0);
                infs1.push_back(inf1);
            }

            // printRecord(odRcd, frameCnt);  // print the record
            cout << "Frame " << frameCnt << ">\t" << "OD&PAR&ACT: " << inf0 << "ms," << "\tFD: " << inf1 << "ms\n";

            frames.clear();
            vchIDs.clear();
            frameCnts.clear();
        }

        if (frameCnt >= frameLimit || endOfFrames)
            break;
    }

    destroyModel();  // destroy all models

    if (infs0.size() > 1) {
        float avgInf0 = accumulate(infs0.begin(), infs0.end(), 0) / infs0.size();
        float avgInf1 = accumulate(infs1.begin(), infs1.end(), 0) / infs1.size();
        cout << "\nAverage>\tOD&PAR&ACT: " << avgInf0 << "ms," << "\tFD: " << avgInf0 << "ms\n";
        cout << "\nModel Configure:";
        cout << "\n  -OD & Tracking: " << cfg.odEnable << "\n  -FD:" << cfg.fdEnable << "\n  -PAR: " << cfg.parEnable
             << "\n  -POSE: " << cfg.poseEnable << "\n  -ACT: " << cfg.actEnable << endl;
    }

    cout << "\nOutput file(s):\n";
    for (auto &outFile : cfg.outputFiles)
        cout << "  -" << outFile << endl;

    cout << "\nTerminate program!\n";

    return 0;
}

void printRecord(ODRecord &odRcd, unsigned int frameCnt) {
#ifdef _WIN32
    HANDLE hStdout;
    COORD destCoord;
    hStdout = GetStdHandle(STD_OUTPUT_HANDLE);

    // position cursor at start of window
    destCoord.X = 0;
    destCoord.Y = 0;
    SetConsoleCursorPosition(hStdout, destCoord);
#endif

    cout << "Zone History               \n";  // spaces needed
    for (int i = 0; i < odRcd.zones.size(); i++) {
        Zone &zone = odRcd.zones[i];
        int cM = zone.curPeople[0][0] + zone.curPeople[0][1] + zone.curPeople[0][2];
        int cF = zone.curPeople[1][0] + zone.curPeople[1][1] + zone.curPeople[1][2];

        cout << "Zone " << zone.zoneID << endl;
        cout << " Cur> M: " << cM << "(" << zone.curPeople[0][0] << ", " << zone.curPeople[0][1] << ", "
             << zone.curPeople[0][2] << "), F: " << cF << "(" << zone.curPeople[1][0] << ", " << zone.curPeople[1][1]
             << ", " << zone.curPeople[1][2] << ")" << endl;

        int hM = zone.hitMap[0][0] + zone.hitMap[0][1] + zone.hitMap[0][2];
        int hF = zone.hitMap[1][0] + zone.hitMap[1][1] + zone.hitMap[1][2];

        cout << " Hit> M: " << hM << "(" << zone.hitMap[0][0] << ", " << zone.hitMap[0][1] << ", " << zone.hitMap[0][2]
             << "), F: " << hF << "(" << zone.hitMap[1][0] << ", " << zone.hitMap[1][1] << ", " << zone.hitMap[1][2]
             << ")" << endl;
    }

    cout << "\nCounting Line History\n";
    for (int i = 0; i < odRcd.cntLines.size(); i++) {
        CntLine &cline = odRcd.cntLines[i];

        int uM = cline.totalUL[0][0] + cline.totalUL[0][1] + cline.totalUL[0][2];
        int uF = cline.totalUL[1][0] + cline.totalUL[1][1] + cline.totalUL[1][2];

        cout << "Counting Line " << cline.clineID << endl;
        cout << " U/L> M: " << uM << "(" << cline.totalUL[0][0] << ", " << cline.totalUL[0][1] << ", "
             << cline.totalUL[0][2] << "), F: " << uF << "(" << cline.totalUL[1][0] << ", " << cline.totalUL[1][1]
             << ", " << cline.totalUL[1][2] << ")" << endl;

        int dM = cline.totalDR[0][0] + cline.totalDR[0][1] + cline.totalDR[0][2];
        int dF = cline.totalDR[1][0] + cline.totalDR[1][1] + cline.totalDR[1][2];

        cout << " D/R> M: " << dM << "(" << cline.totalDR[0][0] << ", " << cline.totalDR[0][1] << ", "
             << cline.totalDR[0][2] << "), F: " << dF << "(" << cline.totalDR[1][0] << ", " << cline.totalDR[1][1]
             << ", " << cline.totalDR[1][2] << ")" << endl;
    }
}

void drawZones(ODRecord &odRcd, Mat &img, int vchID, double alpha) {
    int np[1] = {4};
    cv::Mat layer;

    for (Zone &zone : odRcd.zones) {
        if (zone.vchID == vchID) {
            if (layer.empty())
                layer = img.clone();

            int z = zone.zoneID;
            const Scalar color(colorTable[3 * z], colorTable[3 * z + 1], colorTable[3 * z + 2]);
            fillPoly(layer, {zone.pts}, color);
        }
    }

    if (!layer.empty())
        cv::addWeighted(img, alpha, layer, 1 - alpha, 0, img);
}

void drawBoxes(Config &cfg, ODRecord &odRcd, Mat &img, vector<DetBox> &dboxes, int vchID, double alpha,
               const vector<pair<int, int>> &skelPairs) {
    const string *objNames = cfg.odIDMapping.data();
    time_t now = time(NULL);

    vector<Rect> boxes;
    vector<Scalar> boxesColor;
    vector<bool> emphasizes;
    vector<vector<string>> boxTexts;

    if (DRAW_DETECTION) {
        for (auto &dbox : dboxes) {
            if (dbox.objID >= cfg.numClasses)
                continue;

            Rect box(dbox.x, dbox.y, dbox.w, dbox.h);
            boxes.push_back(box);

            int label = dbox.objID;
            if (dbox.prob < cfg.odScoreTh)
                continue;  // should check scores are ordered. Otherwise, use continue

            Scalar boxColor(50, 255, 255);
            // string objName = objNames[label] + "(" + to_string((int)(dbox.prob * 100 + 0.5)) + "%)";
            string objName =
                to_string(dbox.trackID) + objNames[label] + "(" + to_string((int)(dbox.prob * 100 + 0.5)) + "%)";
            // string objName = to_string(dbox.trackID);

            // char buf[80];
            // tm *curTm = localtime(&dbox.inTime);
            // strftime(buf, sizeof(buf), "Time: %H:%M:%S", curTm);
            // string timeInfo = string(buf);

            vector<string> texts{objName};
            // vector<string> texts{objName, timeInfo};

            if (label == PERSON) {
                string trkInfo;
                int period = now - dbox.inTime;
                if (period < cfg.longLastingObjTh) {  // no action
                    trkInfo = "ET: " + to_string(period) + " (" + to_string((int)dbox.distVar) + ")";
                } else {                              // action (Sleep or Hang around)
                    if (dbox.distVar < cfg.noMoveTh)  // Sleep event
                        trkInfo = "ET: " + to_string(period) + ", No movement(" + to_string((int)dbox.distVar) + ")";
                    else  // Hang around event
                        trkInfo = "ET: " + to_string(period) + ", Hang around(" + to_string((int)dbox.distVar) + ")";
                }
                texts.push_back(trkInfo);

                if (cfg.parEnable && dbox.patts.setCnt != -1) {
                    string genderInfo, ageGroupInfo;
                    bool isFemale;
                    int probFemale;
                    int ageGroup, probAgeGroup;

                    PedAtts::getGenderAtt(dbox.patts, isFemale, probFemale);
                    genderInfo = "Gen: " + string((isFemale ? "F" : "M")) + " (" + to_string(probFemale) + "%)" +
                                 to_string(dbox.patts.setCnt);
                    texts.push_back(genderInfo);

                    boxColor = isFemale ? Scalar(80, 80, 255) : Scalar(255, 80, 80);
                    PedAtts::getAgeGroupAtt(dbox.patts, ageGroup, probAgeGroup);
                    ageGroupInfo =
                        "Age: " +
                        string(ageGroup == CHILD_GROUP ? "child" : (ageGroup == ADULT_GROUP ? "adult" : "elder")) +
                        " (" + to_string(probAgeGroup) + "%)";
                    texts.push_back(ageGroupInfo);
                }

                if (cfg.actEnable && DRAW_ACTION) {
                    if (dbox.actID != -1) {
                        string actInfo;
                        actInfo = "Action: " + cfg.actIDMapping[dbox.actID] + " (" +
                                  to_string((int)(dbox.actConf * 100)) + "%)-" + to_string(dbox.actSetCnt);
                        texts.push_back(actInfo);
                    }
                }
            }

            boxesColor.push_back(boxColor);
            boxTexts.push_back(texts);

            if ((DRAW_CNTLINE && (dbox.justCountedLine > 0)) || (DRAW_ZONE && (dbox.justCountedZone > 0)))
                emphasizes.push_back(true);
            else
                emphasizes.push_back(false);

            if (cfg.poseEnable && DRAW_POSE) {
                Skeleton &skel = dbox.skel;

                if (skel.size() != NUM_SKEL_KEYPOINTS)
                    continue;

                for (auto const &kpt : skel) {
                    float x = kpt.x;
                    float y = kpt.y;
                    if (kpt.confScore >= cfg.poseScoreTh) {
                        cv::circle(img, cv::Point(x, y), 4, cv::Scalar(0, 0, 255), 2, LINE_AA);
                    }
                }
                for (auto const &pair : skelPairs) {
                    SKeyPoint p1 = skel[pair.first - 1];
                    SKeyPoint p2 = skel[pair.second - 1];
                    if (p1.confScore >= cfg.poseScoreTh && p2.confScore >= cfg.poseScoreTh) {
                        cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(0, 255, 0), 2, LINE_AA);
                    }
                }
            }
        }

        Vis::drawBoxes(img, boxes, boxesColor, boxTexts, emphasizes);
    }

    if (DRAW_ZONE)
        drawZones(odRcd, img, vchID, alpha);

    // draw par results
    if (DRAW_ZONE_COUNTING) {
        vector<string> texts = {"People Counting for Each Zone"};

        for (Zone &zone : odRcd.zones) {
            if (vchID == zone.vchID) {
                int curMTotal, curFTotal;
                curMTotal = zone.curPeople[0][0] + zone.curPeople[0][1] + zone.curPeople[0][2];
                curFTotal = zone.curPeople[1][0] + zone.curPeople[1][1] + zone.curPeople[1][2];

                string title = "-Zone " + to_string(zone.zoneID);
                string cur = "  Cur> M: " + to_string(curMTotal) + "(" + to_string(zone.curPeople[0][0]) + ", " +
                             to_string(zone.curPeople[0][1]) + ", " + to_string(zone.curPeople[0][2]) + "), " +
                             " F: " + to_string(curFTotal) + "(" + to_string(zone.curPeople[1][0]) + ", " +
                             to_string(zone.curPeople[1][1]) + ", " + to_string(zone.curPeople[1][2]) + ")";

                int hitMTotal, hitFTotal;
                hitMTotal = zone.hitMap[0][0] + zone.hitMap[0][1] + zone.hitMap[0][2];
                hitFTotal = zone.hitMap[1][0] + zone.hitMap[1][1] + zone.hitMap[1][2];

                string hit = "  Hit> M: " + to_string(hitMTotal) + "(" + to_string(zone.hitMap[0][0]) + ", " +
                             to_string(zone.hitMap[0][1]) + ", " + to_string(zone.hitMap[0][2]) + "), " +
                             " F: " + to_string(hitFTotal) + "(" + to_string(zone.hitMap[1][0]) + ", " +
                             to_string(zone.hitMap[1][1]) + ", " + to_string(zone.hitMap[1][2]) + ")";

                texts.push_back(title);
                texts.push_back(cur);
                texts.push_back(hit);
            }
        }

        Vis::drawTextBlock(img, Point(18, 500), texts, 1, 2);
    }

    if (DRAW_CNTLINE) {
        for (CntLine &cntLine : odRcd.cntLines) {
            if (vchID == cntLine.vchID)
                line(img, cntLine.pts[0], cntLine.pts[1], Scalar(50, 255, 50), 2, LINE_8);
        }
    }

    // draw couniting results
    if (DRAW_CNTLINE_COUNTING) {
        vector<string> texts = {"People Counting for Each Line"};

        for (CntLine &cntLine : odRcd.cntLines) {
            if (vchID == cntLine.vchID) {
                int upMTotal, upFTotal;
                upMTotal = cntLine.totalUL[0][0] + cntLine.totalUL[0][1] + cntLine.totalUL[0][2];
                upFTotal = cntLine.totalUL[1][0] + cntLine.totalUL[1][1] + cntLine.totalUL[1][2];

                string title = "-Counting Line " + to_string(cntLine.clineID);
                string up = "  U/L> M: " + to_string(upMTotal) + "(" + to_string(cntLine.totalUL[0][0]) + ", " +
                            to_string(cntLine.totalUL[0][1]) + ", " + to_string(cntLine.totalUL[0][2]) + "), " +
                            " F: " + to_string(upFTotal) + "(" + to_string(cntLine.totalUL[1][0]) + ", " +
                            to_string(cntLine.totalUL[1][1]) + ", " + to_string(cntLine.totalUL[1][2]) + ")";

                int dwMTotal, dwFTotal;
                dwMTotal = cntLine.totalDR[0][0] + cntLine.totalDR[0][1] + cntLine.totalDR[0][2];
                dwFTotal = cntLine.totalDR[1][0] + cntLine.totalDR[1][1] + cntLine.totalDR[1][2];

                string dw = "  D/R> M: " + to_string(dwMTotal) + "(" + to_string(cntLine.totalDR[0][0]) + ", " +
                            to_string(cntLine.totalDR[0][1]) + ", " + to_string(cntLine.totalDR[0][2]) + "), " +
                            " F: " + to_string(dwFTotal) + "(" + to_string(cntLine.totalDR[1][0]) + ", " +
                            to_string(cntLine.totalDR[1][1]) + ", " + to_string(cntLine.totalDR[1][2]) + ")";

                texts.push_back(title);
                texts.push_back(up);
                texts.push_back(dw);
            }
        }

        Vis::drawTextBlock(img, Point(18, 40), texts, 1, 2);
    }
}

void drawBoxesFD(Config &cfg, FDRecord &fdRcd, Mat &img, vector<FireBox> &fboxes, int vchID) {
    const string *objNames = cfg.fdIDMapping.data();
    time_t now = time(NULL);

    vector<Rect> boxes;
    vector<Scalar> boxesColor;
    vector<vector<string>> boxTexts;

    if (DRAW_FIRE_DETECTION) {
        for (auto &fbox : fboxes) {
            if (fbox.objID >= NUM_CLASSES_FD)  // draw only the fire
                continue;

            Rect box(fbox.x, fbox.y, fbox.w, fbox.h);
            boxes.push_back(box);

            int label = fbox.objID;
            if (fbox.prob < cfg.fdScoreTh)
                continue;  // should check scores are ordered. Otherwise, use continue

            string objName = objNames[label] + "(" + to_string((int)(fbox.prob * 100 + 0.5)) + "%)";

            Scalar boxColor;
            if (label == FIRE)
                boxColor = Scalar(0, 0, 255);
            else
                boxColor = Scalar(220, 200, 200);

            char buf[80];
            tm *curTm = localtime(&now);
            strftime(buf, sizeof(buf), "Time: %H:%M:%S", curTm);
            string timeInfo = string(buf);

            // vector<string> texts{objName};
            vector<string> texts{objName, timeInfo};

            boxesColor.push_back(boxColor);
            boxTexts.push_back(texts);
        }

        Vis::drawBoxesFD(img, boxes, boxesColor, boxTexts);
    }

    if (DRAW_FIRE_DETECTION_COUNTING) {
        int numFire = 0, numSmoke = 0;

        for (auto &fbox : fboxes) {
            if (fbox.prob < cfg.fdScoreTh)
                continue;  // should check scores are ordered. Otherwise, use continue

            if (fbox.objID == FIRE)  // draw only the fire
                numFire++;

            if (fbox.objID == SMOKE)  // draw only the fire
                numSmoke++;
        }

        string fdText = "Detection: Fire=" + to_string(numFire) + ", Smoke=" + to_string(numSmoke);
        Vis::drawTextBlockFD(img, fdRcd, vchID, 40, fdText, 1, 2);
    }
}

bool parseConfigAPI(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, VideoDir &videoDir) {
    string jsonCfgFile = CFG_FILEPATH;
    std::ifstream cfgFile(jsonCfgFile);
    json js;
    cfgFile >> js;

    // apikey, gpu_id
    cfg.frameLimit = js["global"]["frame_limit"];
    cfg.key = js["global"]["apikey"];
    cfg.inputFiles = js["global"]["input_files"].get<vector<string>>();
    cfg.outputFiles = js["global"]["output_files"].get<vector<string>>();

    if (cfg.inputFiles.size() != cfg.outputFiles.size()) {
        cout << "input_files and output_files should be the same size!!";
        return false;
    }

    // read the list of filepaths
    videoDir.init(cfg.inputFiles, cfg.outputFiles);

    cfg.numChannels = videoDir.size();
    cfg.frameWidths = videoDir.getFrameWidths();
    cfg.frameHeights = videoDir.getFrameHeights();
    cfg.fpss = videoDir.getFpss();
    cfg.isMainChannel.resize(cfg.numChannels,
                             true);  // you can select main channels to which action recognition is applied

    // od config
    cfg.odEnable = js["od"]["enable"];
    cfg.netWidth = 960;   // fixed
    cfg.netHeight = 544;  // fixed
    cfg.odScoreTh = js["od"]["score_th"];
    cfg.odIDMapping = {"person"};
    // cfg.odIDMapping = {"person", "bycle", "car", "motorcycle", "airplane", "bus", "train", "truck"};
    cfg.numClasses = cfg.odIDMapping.size();

    // fd config
    cfg.fdEnable = js["fd"]["enable"];
    cfg.fdNetWidth = 960;   // fixed
    cfg.fdNetHeight = 544;  // fixed
    cfg.fdBatchSize = 1;    // fixed(but support from 1 to 4)
    cfg.fdScoreTh = js["fd"]["score_th"];
    cfg.fdIDMapping = {"smoke", "fire"};
    cfg.fdNumClasses = cfg.fdIDMapping.size();
    cfg.fdWindowSize = 32;
    cfg.fdPeriod = 30;

    // tracking
    cfg.longLastingObjTh = 300;
    cfg.noMoveTh = 3.0f;
    cfg.lineEmphasizePeroid = 15;

    // par config
    cfg.parEnable = js["par"]["enable"];
    cfg.parBatchSize = 1;  // fixed
    cfg.parIDMapping = {"gender", "child", "adult", "elder"};
    cfg.numAtts = NUM_ATTRIBUTES;
    cfg.attUpdatePeriod = 15;

    // pose config
    cfg.poseScoreTh = js["pose"]["score_th"];
    cfg.poseModelFile = POSE_MD_FILEPATH;
    cfg.poseBatchSize = 4;

    // action config
    cfg.actScoreTh = js["act"]["score_th"];
    cfg.heatmapScoreTh = 0.25f;
    cfg.actModelFile = ACT_MD_FILEPATH;
    cfg.actIDMapping = aipro_t17;
    cfg.actBatchSize = 1;  // fixed
    cfg.actUpdatePeriod = 12;
    cfg.actLastPeriod = 48;
    cfg.multiPersons = false;  // fixed

    // clip config
    cfg.clipLength = 48;
    cfg.missingLimit = 12;
    cfg.maxNumClips = 8;  // 2 * cfg.poseBatchSize;  // can be set to an arbitrary number

    // counting
    cfg.debouncingTh = 20;

#ifndef _CPU_INFER
    cout << "Do inference using GPU\n";
    cfg.poseEnable = js["pose"]["enable"];
    cfg.actEnable = js["act"]["enable"];

    cfg.odBatchSize = 1;  // from 1 to 4
    cfg.odModelFile = OD_MD_FILEPATH;

    cfg.fdModelFile = FD_MD_FILEPATH;

    cfg.parModelFile = PAR_MD_FILEPATH;
    cfg.parLightMode = true;
#else
    cout << "Do inference using CPU\n";
    cfg.poseEnable = false;
    cfg.actEnable = false;

    cfg.odBatchSize = 1;  // should be 1
    cfg.odModelFile = OD_MD_CPU_FILEPATH;

    cfg.fdModelFile = FD_MD_CPU_FILEPATH;

    cfg.parModelFile = PAR_MD_CPU_FILEPATH;
    cfg.parLightMode = true;
#endif
    if (cfg.parLightMode)
        cfg.lineEmphasizePeroid = min(cfg.lineEmphasizePeroid, cfg.attUpdatePeriod);

    // cntline
    vector<vector<int>> cntLineParams = js["line"]["param"].get<vector<vector<int>>>();

    for (auto &cntLineParam : cntLineParams) {
        CntLine cntLine;

        cntLine.enabled = true;
        cntLine.clineID = cntLineParam[0];
        cntLine.vchID = cntLineParam[1];
        cntLine.pts[0].x = cntLineParam[2];
        cntLine.pts[0].y = cntLineParam[3];
        cntLine.pts[1].x = cntLineParam[4];
        cntLine.pts[1].y = cntLineParam[5];

        if (abs(cntLine.pts[0].x - cntLine.pts[1].x) > abs(cntLine.pts[0].y - cntLine.pts[1].y))
            cntLine.direction = 0;  // horizontal line -> use delta y and count U and D
        else
            cntLine.direction = 1;  // vertical line -> use delta x and count L and R

        for (int g = 0; g < NUM_GENDERS; g++) {         // number of genders
            for (int a = 0; a < NUM_AGE_GROUPS; a++) {  // number of age groups
                cntLine.totalUL[g][a] = 0;
                cntLine.totalDR[g][a] = 0;
            }
        }

        odRcd.cntLines.push_back(cntLine);
    }

    // zones
    vector<vector<int>> zoneParams = js["zone"]["param"].get<vector<vector<int>>>();

    if (zoneParams.size() > 100) {
        cout << "The number of zones should not exceed 100.\n";
        return false;
    }

    for (auto &zoneParam : zoneParams) {
        Zone zone;

        zone.enabled = true;
        zone.zoneID = zoneParam[0];
        zone.vchID = zoneParam[1];
        zone.isRestricted = zoneParam[2] ? true : false;

        for (int i = 0; i < 4; i++) {
            Point pt;
            pt.x = zoneParam[2 * i + 3];
            pt.y = zoneParam[2 * i + 4];
            zone.pts.push_back(pt);
        }

        for (int g = 0; g < NUM_GENDERS; g++) {         // number of genders
            for (int a = 0; a < NUM_AGE_GROUPS; a++) {  // number of age groups
                zone.curPeople[g][a] = 0;
                zone.hitMap[g][a] = 0;
            }
        }

        odRcd.zones.push_back(zone);
    }

    if (cfg.fdEnable) {
        fdRcd.fireProbsMul.resize(cfg.numChannels);
        fdRcd.smokeProbsMul.resize(cfg.numChannels);

        for (int i = 0; i < cfg.numChannels; i++) {
            fdRcd.fireProbsMul[i].resize(cfg.fdWindowSize, 0.0f);
            fdRcd.smokeProbsMul[i].resize(cfg.fdWindowSize, 0.0f);
        }
    }

    return true;
}