OpenDTAM
========

An open source implementation of DTAM
On Ubuntu, you need the qtbase5-dev and libopencv-dev packages

## Build Instructions On Ubuntu 12.04 LTS

For building `OpenDTAM`, here is a brief instruction on Ubuntu 12.04 LTS.

### Install dependencies

#### qtbase5-dev

    sudo apt-add-repository ppa:ubuntu-sdk-team/ppa
    sudo apt-get update
    sudo apt-get install qt5base-dev

#### libopencv-dev

    sudo apt-get install libopencv-dev

#### boost

    sudo apt-get install libboost1.48-all-dev

### Build OpenDTAM

    cd OpenDTAM/Cpp
    mkdir Build
    cmake ..
    make

### Run OpenDTAM

    ./a.out

### Trouble Shooting

You may get problems on the versions of the dependencies, if so you can resolve the problems by installing the required ones according to the messages output by `cmake`.

The `Trajectory_30_seconds` directory may reside in different path in your system, you can modify them in `testprog.cpp` before run `make`.
