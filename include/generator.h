/*==============================================================================
* Copyright 2024 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
#pragma once

// https://github.com/gabime/spdlog/discussions/2813
// hahv: if build fails with _imp_MapViewOfFileNuma2, uncomment the following line
#pragma comment(lib, "OneCore.lib")

// config for export dll (windows) or so (linux)
#ifdef _WIN32
#ifdef GENERATOR_EXPORTS
#define GENERATOR_API __declspec(dllexport)
#else
#define GENERATOR_API __declspec(dllimport)
#endif
#else  // changed for linux jetson
#ifdef GENERATOR_EXPORTS
#define GENERATOR_API __attribute__((visibility("default")))
#else
#define GENERATOR_API
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include "global.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <chrono>

// wrapping getNow functions
GENERATOR_API bool getNowLive(std::vector<cv::Mat> &nowMats);
GENERATOR_API bool getNowOD(std::vector<std::string> &nowOD);
GENERATOR_API bool getNowFD(std::vector<std::string> &nowFD);
GENERATOR_API bool getNowCC(std::vector<std::string> &nowCC);

// wrapping Logger
GENERATOR_API void setWriteLogger(Config &cfg);
GENERATOR_API bool initLogger(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);
GENERATOR_API bool checkCmdLogger(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);
GENERATOR_API bool needToDrawLogger(int vchID);
GENERATOR_API void writeDataLogger(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd, cv::Mat &frame,
                                   unsigned int &frameCnt, int vchID, std::chrono::system_clock::time_point now);
GENERATOR_API void destroyLogger();

// wrapping Streamer
GENERATOR_API bool initStreamer(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);
GENERATOR_API bool isEmptyStreamer(int vchID);
GENERATOR_API bool tryPopStreamer(CMat &cmat, int vchID);
GENERATOR_API void writeStreamer(cv::Mat &frame, int vchID);
GENERATOR_API int getPeriodStreamer(int vchID);
GENERATOR_API int getUnsafeSizeStreamer(int vchID);
GENERATOR_API int getUnsafeSizeMaxStreamer();
GENERATOR_API void destroyStreamer();

// parsing
GENERATOR_API bool parseConfigAPI(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);

/** @brief Initialize model
 *
 * @param cfg configuration struct
 * @param odRcd object detection record struct
 * @param fdRcd fire detection record struct
 * @param ccRcd crowd counter record struct
 * @return initialization result(true: success, false: fail)
 */
GENERATOR_API bool initModel(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);

/** @brief Run detection and PAR models for a single frame
 *
 * @param dboxes return detected dboxes of the vchID channel
 * @param frame input frame
 * @param vchID vchID of the input frame
 * @param frameCnt frameCnt of the input frame
 * @param odScoreTh threshold for filtering low confident detections
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModel(std::vector<DetBox> &dboxes, cv::Mat &frame, int vchID, uint frameCnt, float odScoreTh);

/** @brief Run detection and PAR models for a frame batch
 *
 * @param dboxesMulP return detected dboxes of all video channels(vchIDs)
 * @param frames batch of frames
 * @param vchIDs vchIDs of batched frames
 * @param frameCnts frameCnts of batched frames
 * @param odScoreTh threshold for filtering low confident detections
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModel(std::vector<std::vector<DetBox>> &dboxesMul, std::vector<cv::Mat> &frames,
                            std::vector<int> &vchIDs, std::vector<uint> &frameCnts, float odScoreTh);

/** @brief Run fire detection for a single frame
 *
 * @param frame input frame
 * @param vchID vchID of the input frame
 * @param frameCnt frameCnt of the input frame
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModelFD(cv::Mat &frame, int vchID, uint &frameCnt);

/** @brief Run crowd counter for a single frame
 *
 * @param density return the density of people
 * @param frame input frame
 * @param vchID vchID of the input frame
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModelCC(cv::Mat &density, cv::Mat &frame, int vchID);

/** @brief Destroy all models
 *
 * @param None
 * @return flag for the destruction result(true: success, false: fail)
 */
GENERATOR_API bool destroyModel();

/** @brief Reset CntLine and Zone configuration
 *
 * @param odRcd record struct
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLineAndZone(ODRecord &odRcd);

/** @brief Reset CntLine configuration
 *
 * @param cntLines vector of CntLines
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLine(std::vector<CntLine> &cntLines);

/** @brief Reset Zone configuration
 *
 * @param zones vector of Zones
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetZone(std::vector<Zone> &zones);

/** @brief Reset CntLine and Zone records
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLineAndZoneRecord();

/** @brief Reset CntLine record
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLineRecord();

/** @brief Reset Zone record
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetZoneRecord();

/** @brief Reset FD
 *
 * @param odRcd record struct
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetFD(FDRecord *pFDRcd);

/** @brief Reset FD record
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetFDRecord();
