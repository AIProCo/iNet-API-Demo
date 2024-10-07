#include "camerastreamer.hpp"

#include <string>

#include "opencv2/opencv.hpp"
//#include "util.h"

CameraStreamer::CameraStreamer(Config &cfg, vector<ODRecord> &odRcds, vector<CCRecord> &ccRcds, Logger *_pLogger)
    : DebugMessage(cfg) {
    pOdRcds = &odRcds;
    pCcRcds = &ccRcds;
    pLogger = _pLogger;

    numChannels = cfg.numChannels;
    maxBufferSize = cfg.maxBufferSize;

    frameCnts.resize(numChannels, 0);

    inputs = cfg.inputFiles;
    outputs = cfg.outputFiles;

    stopFlag = false;

    if (cfg.boostMode)
        initSleepPeriod = 95;  // for 10 fps
    else
        initSleepPeriod = 190;  // for 5 fps

    sleepPeriods.resize(numChannels, initSleepPeriod);

    videoWriters.resize(numChannels);
    cmatsAll.resize(numChannels);

    for (int vchID = 0; vchID < numChannels; vchID++) {
        if (stopFlag)
            return;

        std::thread *t = new std::thread(&CameraStreamer::keepConnected, this, vchID);
        cameraThreads.push_back(t);
    }
}

CameraStreamer::~CameraStreamer() {
    // destroy();
}

void CameraStreamer::destroy() {
    stopFlag = true;
    // std::cout << "destory CameraStreamer Start\n";

    for (const auto &t : cameraThreads) {
        t->join();
        delete t;
    }

    if (pCfg->recording)
        for (auto &videoWriter : videoWriters)
            videoWriter.release();

    // std::cout << "destory CameraStreamer ends\n";
}

void CameraStreamer::keepConnected(int vchID) {
    std::string input = inputs[vchID];
    bool firstOpen = true;

    if (input.empty() || input.length() < 5) {
        lg(std::format("[{}] Wrong address and close this channel: {}\n", vchID, input));
        return;
    }

    while (true) {
        if (stopFlag)
            return;

        time_t time_begin = time(0);
        cv::VideoCapture capture(input);  // opencv capture 객체 생성

        time_t time_1 = time(0) - time_begin;

        if (capture.isOpened()) {        //연결 완료
            pCfg->vchStates[vchID] = 1;  // set connected flag

            // Quit before conntect
            if (stopFlag)
                return;

            lg(std::format("[{}] Open: {}\n", vchID, input));
            // std::cout << "[" << vchID << "] Opened: " << input << std::endl;

            if (firstOpen) {
                firstOpen = false;

                int frameWidth = capture.get(CAP_PROP_FRAME_WIDTH);
                int frameHeight = capture.get(CAP_PROP_FRAME_HEIGHT);
                float fps = capture.get(CAP_PROP_FPS);

                if (input.length() > 5) {
                    if (tolower(input[0]) == 'r' && tolower(input[1]) == 't' && tolower(input[2]) == 's' &&
                        tolower(input[3]) == 'p' && input[4] == ':') {
                        if (pCfg->boostMode)
                            fps = 10.0f;
                        else
                            fps = 5.0f;
                    } else {
                        fps = min(30.0f, max(1.0f, fps));
                    }
                }

                pCfg->frameHeights[vchID] = frameHeight;
                pCfg->frameWidths[vchID] = frameWidth;
                pCfg->fpss[vchID] = fps;

                if (frameHeight < 0 || frameHeight > 2160 || frameWidth < 0 || frameWidth > 3840) {
                    lg(std::format("[{}] Unsupported bitstream: {} {}", vchID, frameHeight, frameWidth));
                    exit(-1);
                }

                for (CntLine &c : (*pOdRcds)[vchID].cntLines) {
                    if (vchID == c.vchID) {
                        if (c.pts[0].x < 0 || c.pts[0].x >= frameWidth || c.pts[1].x < 0 || c.pts[1].x >= frameWidth) {
                            lg(std::format("[{}] cntLine pt.x error in keepConnected: {} {} {}", vchID, c.pts[0].x,
                                           c.pts[1].x, frameWidth));
                            exit(-1);
                        }
                        if (c.pts[0].y < 0 || c.pts[0].y >= frameHeight || c.pts[1].y < 0 ||
                            c.pts[1].y >= frameHeight) {
                            lg(std::format("[{}] cntLine pt.y error in keepConnected: {} {} {}", vchID, c.pts[0].y,
                                           c.pts[1].y, frameHeight));
                            exit(-1);
                        }
                    }
                }

                for (Zone &z : (*pOdRcds)[vchID].zones) {
                    if (vchID == z.vchID) {
                        for (Point &pt : z.pts) {
                            if (pt.x < 0 || pt.x >= frameWidth) {
                                lg(std::format("[{}] zone pt.x error in keepConnected: {} {}", vchID, pt.x,
                                               frameWidth));
                                exit(-1);
                            }

                            if (pt.y < 0 || pt.y >= frameHeight) {
                                lg(std::format("[{}] zone pt.y error in keepConnected: {} {}", vchID, pt.y,
                                               frameHeight));
                                exit(-1);
                            }
                        }
                    }
                }

#ifndef _CPU_INFER
                for (CCZone &z : (*pCcRcds)[vchID].ccZones) {
                    if (vchID == z.vchID) {
                        for (Point &pt : z.pts) {
                            if (pt.x < 0 || pt.x >= frameWidth) {
                                lg(std::format("[{}] ccZone pt.x error in keepConnected: {} {}", vchID, pt.x,
                                               frameWidth));
                                exit(-1);
                            }

                            if (pt.y < 0 || pt.y >= frameHeight) {
                                lg(std::format("[{}] ccZone pt.y error in keepConnected: {} {}", vchID, pt.y,
                                               frameHeight));
                                exit(-1);
                            }
                        }
                    }
                }
#endif
                pCfg->odScaleFactors[vchID] =
                    std::min((float)pCfg->odNetWidth / frameWidth, (float)pCfg->odNetHeight / frameHeight);
                pCfg->odScaleFactorsInv[vchID] = 1.0f / pCfg->odScaleFactors[vchID];

                pCfg->fdScaleFactors[vchID] =
                    std::min((float)pCfg->fdNetWidth / frameWidth, (float)pCfg->fdNetHeight / frameHeight);
                pCfg->fdScaleFactorsInv[vchID] = 1.0f / pCfg->fdScaleFactors[vchID];

                pCfg->ccScaleFactors[vchID] =
                    std::min((float)pCfg->ccNetWidth / frameWidth, (float)pCfg->ccNetHeight / frameHeight);
                pCfg->ccScaleFactorsInv[vchID] = 1.0f / pCfg->ccScaleFactors[vchID];

                if (pCfg->recording)
                    videoWriters[vchID].open(outputs[vchID], VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                                             Size(frameWidth, frameHeight));  ///*.mp4 format

                lg(std::format("[{}] ({}, {}), {}\n", vchID, frameWidth, frameHeight, fps));

                pLogger->writeChInfo();
            }

            bool response = working(&capture, vchID);
            if (response == false) {
                pCfg->vchStates[vchID] = 0;  // set not-connected

                if (stopFlag)
                    return;

                // Wait for 2 sec and try to reconnect (keep connected with
                // real-time ip camera)
                universal_sleep(2000);
            }
        } else {
            std::cout << "[" << vchID << "] Can't connect: " << input << std::endl;
            universal_sleep(5000);
        }
    }
}

bool CameraStreamer::working(cv::VideoCapture *capture, int vchID) {
    int grapFailCnt = 0;
    int retrieveEmptyCnt = 0;

    while (true) {
        int &sleepPeriod = sleepPeriods[vchID];

        if (!capture->grab()) {
            grapFailCnt++;
            lg(std::format(" [{}]Grab Fail - Fail Count: {}\n", vchID, grapFailCnt));

            if (grapFailCnt > 2)
                return false;
        } else {
            cv::Mat frame;
            capture->retrieve(frame);  // capture 객체를 이용한 영상 수신
            if (frame.empty()) {
                retrieveEmptyCnt++;
                lg(std::format(" [{}]Retrieve Fail - Fail Count: {}\n", vchID, retrieveEmptyCnt));

                if (retrieveEmptyCnt > 2)
                    return false;
            } else {
                CMats &cmats = cmatsAll[vchID];
                int curBufferSize = cmats.unsafe_size();

                if (curBufferSize >= maxBufferSize / 2) {
                    if (sleepPeriod < 3 * initSleepPeriod)
                        sleepPeriod += 5;

                    if (curBufferSize >= maxBufferSize) {
                        CMat tmp;
                        cmats.try_pop(tmp);
                    }
                } else {
                    if (sleepPeriod > initSleepPeriod)
                        sleepPeriod -= 5;
                }

                int frameCnt = frameCnts[vchID]++;
                cmats.push(CMat(frame, vchID, frameCnt));
            }
        }

        if (stopFlag)
            return false;

        universal_sleep(sleepPeriod);
    }
    return true;
}
