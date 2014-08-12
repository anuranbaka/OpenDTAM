#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <utils/ImplThreadLaunch.hpp>
void pfShow(const std::string name,const cv::Mat& _mat, int defaultscale=0,cv::Vec2d autoscale=cv::Vec2d(0,0));
void pfWindow(const std::string name,int prop);
void* guiLoop(void*);
void initGui();
void gpause();
void gcheck();
extern int allDie;
#endif
