template <class T, class PixelType>// T is the datatype of the cost volume, PixelType is the image 
class Cost{
public:
    Mat_<float> hits;// number of times a cell has been hit
    const Mat_<PixelType> baseImage;
    int rows;
    int cols;
    int layers;
    const Matx33d cameraMatrix;
    const Matx44d pose;//the affine transform representing the world -> camera frame transformation
    T* data;// stores the [rows][cols][layers] array 
    Vector<float> depth;
    
    Cost(int rows,int cols,int layers, Mat& cameraMatrix, Mat& cameraPose);
    
    updateCost(Mat& image, Matx44d& currentCameraPose);
    
    
    
    
    
private:
    Mat_<T> container; //stores the actual data for data*
    
    T photoCost(PixelType& a,PixelType& b)
    void reproject( const Mat_<PixelType>& src
                    const Matx33d& cameraMatrix,
                    const Matx44d& cameraAffinePoseBase,
                    const Matx44d& cameraAffinePoseAlternate,
                    float e,
                    Mat_<PixelType>& dst,
                    Mat& mask);
    
};
