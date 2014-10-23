// loads all files of a given name and extension
#include "fileLoader.hpp"
    #include <boost/filesystem.hpp>
    
    using namespace cv;
    using namespace std;
    namespace fs = ::boost::filesystem;
    static fs::path root;
static vector<fs::path> txt;
static vector<fs::path> png;
static vector<fs::path> depth;


void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret);
void loadAhanda(const char * rootpath,
                int imageNumber,
                Mat& image,
                Mat& d,
                Mat& cameraMatrix,
                Mat& R,
                Mat& T){
    if(root!=string(rootpath)){

        root=string(rootpath);
        get_all(root, ".txt", txt);
        get_all(root, ".png", png);
        get_all(root, ".depth", depth);
                cout<<"Loading"<<endl;
    }
    
    convertAhandaPovRayToStandard(txt[imageNumber].c_str(),
                                      cameraMatrix,
                                      R,
                                      T);
    cout<<"Reading: "<<png[imageNumber].filename().string()<<endl;
    imread(png[imageNumber].string(), -1).convertTo(image,CV_32FC3,1.0/65535.0,1/255.0);
if(depth.size()>0){
    
}
        
}








































#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 


// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret)
{  
  if (!fs::exists(root)) return;

  if (fs::is_directory(root))
  {
    typedef std::set<boost::filesystem::path> Files;
    Files files;
    fs::recursive_directory_iterator it0(root);
    fs::recursive_directory_iterator endit0;
    std::copy(it0, endit0, std::inserter(files, files.begin()));
    Files::iterator it= files.begin();
    Files::iterator endit= files.end();
    while(it != endit)
    {
      if (fs::is_regular_file(*it) && (*it).extension() == ext)
      {
        ret.push_back(*it);
      }
      ++it;
    }
  }
}