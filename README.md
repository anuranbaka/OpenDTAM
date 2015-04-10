OpenDTAM
========

An open source implementation of DTAM

Based on Newcombe, Richard A., Steven J. Lovegrove, and Andrew J. Davison's "DTAM: Dense tracking and mapping in real-time."

This project depends on qtbase5-dev and [OpenCV 3](https://github.com/Itseez/opencv "OpenCV").

## Build Instructions on Ubuntu 12/14

### Install dependencies

#### qtbase5-dev

```bash
sudo apt-add-repository ppa:ubuntu-sdk-team/ppa
sudo apt-get update
sudo apt-get install qtbase5-dev
```

#### OpenCV 3

```bash
# Execute first command from directory you would like to clone OpenCV
git clone https://github.com/Itseez/opencv
cd opencv
mkdir build
cd build
cmake ..
make -j4
sudo make install
```

#### boost

```bash
sudo apt-get install libboost-system-dev libboost-thread-dev
```

### Build OpenDTAM
```bash
cd OpenDTAM
mkdir build
cd build
cmake ../Cpp
make -j4
````

### Run OpenDTAM
For the following command, replace `$TRAJECTORY_30_SECONDS` with the path to the directory of the same name.
```bash
./a.out $TRAJECTORY_30_SECONDS
```
Assuming you are executing this command from the build folder, as shown above, enter the following:
```bash
./a.out ../Trajectory_30_seconds
```

### Trouble Shooting

You may have problems with the versions of the dependencies, if so you may be able to resolve them by installing the required ones according to the messages output by `cmake`.

If you encounter errors, send me an email with the output of cmake (something like " -- Detected version of GNU GCC: 46 (406) .......")  
and also the output of   
"cmake -L"  
and I can tell you what you need to do.
