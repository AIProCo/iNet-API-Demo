#include <string>

#include "CameraStreamer.hpp"
#include "util.h"
#include "opencv2/opencv.hpp"

CameraStreamer::CameraStreamer(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd) {
    pCfg = &cfg;
    pOdRcd = &odRcd;
    pFdRcd = &fdRcd;
    pCcRcd = &ccRcd;

    lg = cfg.lg;

    numChannels = cfg.numChannels;
    maxBufferSize = cfg.maxBufferSize;

    frameCnts.resize(numChannels, 0);

    inputs = cfg.inputFiles;
    outputs = cfg.outputFiles;

    stopFlag = false;

    if (cfg.boostMode)
        initSleepPeriod = 90;  // for 10 fps
    else
        initSleepPeriod = 180;  // for 5 fps

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

void CameraStreamer::destory() {
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

    while (true) {
        if (stopFlag)
            return;

        time_t time_begin = time(0);
        cv::VideoCapture capture(input);
        // cv::VideoCapture capture(url, cv::CAP_DSHOW);

        // bool tmp;
        // tmp = capture->set(cv::CAP_PROP_BUFFERSIZE, 1);
        // std::cout << "tmp = " << tmp << std::endl;

        // std::string bname = capture->getBackendName();
        // std::cout << "bname = " << bname << std::endl;

        time_t time_1 = time(0) - time_begin;
        // std::cout << "[" << vchID << "] Delay: " << time_1 << " -> ";

        if (capture.isOpened()) {
            pCfg->vchStates[vchID] = 1;  // set connected

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

                pCfg->frameHeights[vchID] = frameHeight;
                pCfg->frameWidths[vchID] = frameWidth;
                pCfg->fpss[vchID] = fps;

                if (frameHeight < 0 || frameHeight > 2160 || frameWidth < 0 || frameWidth > 3840) {
                    lg(std::format("Unsupported bitstream: {} {}", frameHeight, frameWidth));
                    exit(-1);
                }

                for (CntLine &c : pOdRcd->cntLines) {
                    if (vchID == c.vchID) {
                        if (c.pts[0].x < 0 || c.pts[0].x >= frameWidth || c.pts[1].x < 0 || c.pts[1].x >= frameWidth) {
                            lg(std::format("cntLine pt.x error in keepConnected: {} {} {}", c.pts[0].x, c.pts[1].x,
                                           frameWidth));
                            exit(-1);
                        }
                        if (c.pts[0].y < 0 || c.pts[0].y >= frameHeight || c.pts[1].y < 0 ||
                            c.pts[1].y >= frameHeight) {
                            lg(std::format("cntLine pt.y error in keepConnected: {} {} {}", c.pts[0].y, c.pts[1].y,
                                           frameHeight));
                            exit(-1);
                        }
                    }
                }

                for (Zone &z : pOdRcd->zones) {
                    if (vchID == z.vchID) {
                        for (Point &pt : z.pts) {
                            if (pt.x < 0 || pt.x >= frameWidth) {
                                lg(std::format("zone pt.x error in keepConnected: {} {}", pt.x, frameWidth));
                                exit(-1);
                            }

                            if (pt.y < 0 || pt.y >= frameHeight) {
                                lg(std::format("zone pt.y error in keepConnected: {} {}", pt.y, frameHeight));
                                exit(-1);
                            }
                        }
                    }
                }

#ifndef _CPU_INFER
                for (CCZone &z : pCcRcd->ccZones) {
                    if (vchID == z.vchID) {
                        for (Point &pt : z.pts) {
                            if (pt.x < 0 || pt.x >= frameWidth) {
                                lg(std::format("ccZone pt.x error in keepConnected: {} {}", pt.x, frameWidth));
                                exit(-1);
                            }

                            if (pt.y < 0 || pt.y >= frameHeight) {
                                lg(std::format("ccZone pt.y error in keepConnected: {} {}", pt.y, frameHeight));
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

                lg(std::format("vch {}: ({}, {}), {}\n", vchID, frameWidth, frameHeight, fps));
                // cout << std::format("vch {}: ({}, {}), {}\n", vchID, frameWidth, frameHeight, fps);
            }

            bool response = working(&capture, vchID);
            if (response == false) {
                pCfg->vchStates[vchID] = 0;  // set not-connected

                if (stopFlag)
                    return;

                // Wait for 2 sec and try to reconnect (keep connected with real-time ip camera)
                universal_sleep(2000);
            }
        } else {
            std::cout << "[" << vchID << "] Can't connect: " << input << std::endl;
            universal_sleep(5000);
        }
    }
};

// bool CameraStreamer::working(cv::VideoCapture* capture, int index, con_queue* input_q) { //for video file
//	unsigned int cnt = 0;
//
//	while (true) {
//		if (stopFlag) return false;
//
//		if (!capture->grab()) {
//			std::cout << "Fail to grab frame : " << (index+1) << " camera" << std::endl;
//			return false;
//		}
//		else {
//			cv::Mat frame;
//			capture->retrieve(frame);
//			if (frame.empty()) {
//				std::cout << "Frame is empty, so pass it : " << (index+1) << " camera" << std::endl;
//				return false;
//			}
//			else {
//				if(input_q->unsafe_size() > MAX_QUEUE_SIZE) {
//					if(sleep_period_cs < MAX_SLEEP_TIME)
//						++sleep_period_cs;
//
//					Sleep(sleep_period_cs);
//				}
//				else if(sleep_period_cs > 0)
//					sleep_period_cs--;
//
//				input_q->push(std::make_pair(frame, index));
//			}
//		}
//	}
//	return true;
//}

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
            capture->retrieve(frame);
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
