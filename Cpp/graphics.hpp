#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP
#include <string>
#include <opencv2/core/mat.hpp>
void pfShow(const std::string name,const cv::Mat& _mat);
void pfWindow(const std::string name,int prop);
void* guiLoop(void*);
void initGui();
#endif