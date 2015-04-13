OpenDTAM
========

An open source implementation of DTAM

Based on Newcombe, Richard A., Steven J. Lovegrove, and Andrew J. Davison's "DTAM: Dense tracking and mapping in real-time."

On Ubuntu, you need the qtbase5-dev and libopencv-dev packages

## Build Instructions On Ubuntu 12.04 LTS

*Notice:* This repo now tracks my OpenCV repo which is a fork of OpenCV 3.0. Unless you're a real masochist, you want to use the 2.4.9_backport branch. I will do my best to keep that as a easy to compile branch.

For building `OpenDTAM`, here is a brief instruction on Ubuntu 12.04 LTS.

### Install dependencies

#### qtbase5-dev

    sudo apt-add-repository ppa:ubuntu-sdk-team/ppa
    sudo apt-get update
    sudo apt-get install qtbase5-dev

#### libopencv-dev

    sudo apt-get install libopencv-dev

#### boost

    sudo apt-get install libboost1.48-all-dev

#### CUDA

     sudo add-apt-repository ppa:ubuntu-x-swat/x-updates
     sudo apt-get update
     sudo apt-get install nvidia-current nvidia-cuda-dev

### Build OpenDTAM

    cd OpenDTAM/Cpp
    mkdir Build
    cd Build
    cmake ..
    make

### Run OpenDTAM

    ./a.out

### Trouble Shooting

Running  "pkg-config --modversion opencv" will tell you what version you have. Hopefully it is 
close to 2.4.9 if you're on the backport branch. All bets are off if you're on the master.

You may have problems with the versions of the dependencies, if so you may be able to resolve them by installing the required ones according to the messages output by `cmake`.

The `Trajectory_30_seconds` directory may reside in different path in your system, you can modify them in `testprog.cpp` before running `make`.

Or if none of this works compile OpenCV 2.4.9 from source on your machine.  
Then send me an email with the output of cmake (something like " -- Detected version of GNU GCC: 46 (406) .......")  
and also the output of   
"cmake -L"  
and I can tell you what you need to do.
