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

// parsing
GENERATOR_API bool parseConfigAPI(Config &cfg, std::vector<ODRecord> &odRcds, std::vector<FDRecord> &fdRcds,
                                  std::vector<CCRecord> &ccRcds, std::vector<MinObj> &minObjs);

/** @brief Initialize model
 *
 * @param cfg configuration struct
 * @param odRcd object detection record struct
 * @param fdRcd fire detection record struct
 * @param ccRcd crowd counter record struct
 * @return initialization result(true: success, false: fail)
 */
GENERATOR_API bool initModel(Config &cfg);

/** @brief Run detection and PAR models for a single frame
 *
 * @param dboxes return detected dboxes of the vchID channel
 * @param odRcd object detection record struct
 * @param minObj minimum size object deletion struct
 * @param frame input frame
 * @param vchID vchID of the input frame
 * @param frameCnt frameCnt of the input frame
 * @param odScoreTh threshold for filtering low confident detections
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModel(std::vector<DetBox> &dboxes, ODRecord &odRcd, MinObj &minObj, cv::Mat &frame, int vchID,
                            uint frameCnt, float odScoreTh);

/** @brief Run detection and PAR models for a frame batch
 *
 * @param fdRcd fire detection record struct
 * @param frame input frame
 * @param vchID vchID of the input frame
 */
GENERATOR_API bool runModelFD(FDRecord &fdRcd, cv::Mat &frame, int vchID);

/** @brief Run crowd counter for a single frame
 *
 * @param density return the density of people
 * @param frame input frame
 * @param vchID vchID of the input frame
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModelCC(cv::Mat &density, CCRecord &ccRcd, cv::Mat &frame, int vchID);

/** @brief Destroy all models
 *
 * @param None
 * @return flag for the destruction result(true: success, false: fail)
 */
GENERATOR_API bool destroyModel();