OpenDTAM
========

An open source implementation of DTAM
On Ubuntu, you need the qtbase5-dev and libopencv-dev packages

## Build Instructions On Ubuntu 12.04 LTS

*Notice:* From now until 8/18/14 I will be attempting to get this code to compile on OpenCV 3.0.0
Getting this to build from the latest commits will therefore be nigh impossible.
Instead try commit a5f91d0cd58c3353ef56a3694648375f1c053082 on the main branch and use OpenCV 2.4.something

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
close to 2.4.9 which is what commit a5f91d0cd58c3353ef56a3694648375f1c053082  was built for.

You may have problems with the versions of the dependencies, if so you may be able to resolve them by installing the required ones according to the messages output by `cmake`.

The `Trajectory_30_seconds` directory may reside in different path in your system, you can modify them in `testprog.cpp` before running `make`.

Or if none of this works compile a version of OpenCV from source on your machine.  
Then send me an email with the output of cmake (something like " -- Detected version of GNU GCC: 46 (406) .......")  
and also the output of   
"cmake -L"  
and I can tell you what you need to do.
