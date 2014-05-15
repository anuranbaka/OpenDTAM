// The cost volume. Conceptually arranged as an image plane, corresponding 
// to the keyframe, lying on top of the actual cost volume, a 3D two channel matrix storing 
// the total cost of all rays that have passed through a voxel, and the number of rays that 
// have hit that voxel.
//
// There is also the depth array, which stores the inverse depths of each plane of the cost.
//
// For efficiency, the accumulated cost and hit count are seperate arrays even though 
// conceptually they are just different channels of the same image.
//
// The cost volume doesn't support updating by a different camera than the one that took the 
// keyframe, because that would violate a bunch of assumptions for DTAM


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
    T* data;// stores the [rows][cols][layers] array of sum of costs so far
    T* hit;//stores the number of times each cell has been hit by a ray
    const Vector<float> depth;
    
    Cost(int rows,int cols,int layers, Mat& cameraMatrix, Mat& cameraPose);// autogenerate default depths
    Cost(int rows,int cols,const Vector<float>& depth, Mat& cameraMatrix, Mat& cameraPose);//use given depths
    
    updateCost(Mat& image, Matx44d& currentCameraPose);
    
    
    
    
    
private:
    Mat_<T> dataContainer; //stores the actual data for data*, used for auto allocation behavior
    Mat_<T> hitContainer; //stores the actual data for hit*, used for auto allocation behavior
    
    T photoCost(PixelType& a,PixelType& b)
    void reproject( const Mat_<PixelType>& src
                    const Matx33d& cameraMatrix,
                    const Matx44d& cameraAffinePoseBase,
                    const Matx44d& cameraAffinePoseAlternate,
                    float invDepth,
                    Mat_<PixelType>& dst,
                    Mat& mask);
    Vector<float> generateDepths(int layers);// generate the depth list, the signature will probably change
    
};
