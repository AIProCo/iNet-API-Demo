cmake_minimum_required(VERSION 3.10)

if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(ColourBold "${Esc}[1m")
    set(Red "${Esc}[31m")
    set(Green "${Esc}[32m")
    set(Yellow "${Esc}[33m")
    set(Blue "${Esc}[34m")
    set(Magenta "${Esc}[35m")
    set(Cyan "${Esc}[36m")
    set(White "${Esc}[37m")
    set(BoldRed "${Esc}[1;31m")
    set(BoldGreen "${Esc}[1;32m")
    set(BoldYellow "${Esc}[1;33m")
    set(BoldBlue "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan "${Esc}[1;36m")
    set(BoldWhite "${Esc}[1;37m")
endif()

# set device inference type; default is GPU; set BUILD_FOR_CPU to ON to build for CPU
# set (BUILD_FOR_CPU ON)
set(COLORED_LOG ON) # set COLORED_LOG to OFF to disable colored log

if(BUILD_FOR_CPU)
    add_definitions(-D_CPU_INFER) # define _CPU_INFER for buid process
    message(STATUS "${BoldRed} ----> BUILD for CPU <----${ColourReset}")
else()
    message(STATUS "${BoldRed} ----> BUILD for GPU <----${ColourReset}")
endif()

if(COLORED_LOG)
    add_definitions(-D_COLORED_LOG) # define COLORED_LOG for buid process
    message(STATUS "${BoldRed} ----> COLORED LOG <----${ColourReset}")
endif()

project(client)

# https://stackoverflow.com/questions/1620918/cmake-and-libpthread
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the C compiler
set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")

# Set optimization flags for GCC and Clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "GCC or Clang detected, setting optimization flags To -O2: Optimized for Speed")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

# Enable AddressSanitizer only for Debug builds
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "AddressSanitizer enabled for Debug build")
        set(ADDRESS_SANITIZER_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
        set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
    endif()
endif()

find_package(Threads REQUIRED)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the source files for the client executable
file(GLOB SOURCES "*.cpp")
# Add executable target
add_executable(client ${SOURCES})

message("--------------Header Include Directories----------------")
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLIENT_INCLUDE_DIR ${PROJECT_ROOT_DIR}/include)
set(OPENCV_INCLUDE_DIR  ${PROJECT_ROOT_DIR}/include_opencv)
# tbb is used for concurrent_queue.h
set(TBB_INCLUDE_DIR /usr/include/oneapi)

# Add the include directories
target_include_directories(client PUBLIC ${CLIENT_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR} ${TBB_INCLUDE_DIR})

# Print include directories
get_target_property(dirs client INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message("-- ${dir}")
endforeach()

message("--------------Lib Link Directories----------------")

set(LIB_DIR ${PROJECT_ROOT_DIR}/bin)
message("-- Linux Lib DIR= ${LIB_DIR}")
set(TBB_LIB_DIR /usr/lib/x86_64-linux-gnu)

# add the link directories (where the libraries are)
target_link_directories(client PUBLIC ${LIB_DIR} ${TBB_LIB_DIR} ${CMAKE_THREAD_LIBS_INIT})

set(OPENCV_DEP_LIB0 libopencv_world.so)
set(TBBLIB_DEP_LIB0 libtbb.so)

# find all .so files in the lib directory
file(GLOB LIBS "${LIB_DIR}/*.so")

# add the libraries to the target, depending on type of build (CPU or GPU)
foreach(lib ${LIBS})
    message("-- Adding lib: ${lib}")

    if(lib MATCHES "generator")
        if(BUILD_FOR_CPU)
            target_link_libraries(client PUBLIC "${LIB_DIR}/libgenerator_cpu.so")
        else()
            target_link_libraries(client PUBLIC "${LIB_DIR}/libgenerator.so")
        endif()
    else()
        target_link_libraries(client PUBLIC ${lib})
    endif()
endforeach()

# target_link_libraries(client PUBLIC generator)
target_link_libraries(client PUBLIC Threads::Threads)

# Add the libraries (lib names used, i.e.: .so files)
# target_link_libraries(client PUBLIC ${OPENCV_DEP_LIB0})
# target_link_libraries(client PUBLIC ${TBBLIB_DEP_LIB0})


get_target_property(link_directories client LINK_DIRECTORIES)
foreach(dir ${link_directories})
    message("-- ${dir}")
endforeach()

message("----------------------------------------------")

message("--------------POST BUILD COMMANDS----------------")
# # Copy client exce file to the final release folder
# Add commands to copy the built executables and shared libraries to common/bin directory

message("-- Copying <client> exec to ${LIB_DIR}")
add_custom_command(TARGET client POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:client> ${LIB_DIR}
)

