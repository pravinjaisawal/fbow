#include <iostream>
#include <fstream>
#include <vector>
#include <map>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

// fov headers
#include "fbow.h"
#include "vocabulary_creator.h"

// directory reader
#include "dirreader.h"


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( std::string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( std::string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } std::string operator()(std::string param,std::string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( std::string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};

std::vector< cv::Mat  >  loadFeatures( std::vector<std::string> path_to_images,std::string descriptor="")  {
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create(2000);
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,  0,  3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    std::vector<cv::Mat>    features;


    std::cout << "Extracting   features..." << std::endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::cout<<"reading image: "<< path_to_images[i]<<std::endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty()) {
            std::cerr<<"Could not open image:"<<path_to_images[i]<<std::endl;
            continue;
        }
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        std::cout<<"extracting features: total= "<<keypoints.size() <<std::endl;
        features.push_back(descriptors);
        std::cout<<"done detecting features"<<std::endl;
    }
    return features;
}


// ----------------------------------------------------------------------------
void saveToFile(std::string filename,const std::vector<cv::Mat> &features,  std::string  desc_name,bool rewrite =true){

    //test it is not created
    if (!rewrite){
        std::fstream ifile(filename, std::ios::binary);
        if (ifile.is_open())//read size and rewrite
            std::runtime_error( "ERROR::: Output File "+filename+" already exists!!!!!" );
    }
	std::ofstream ofile(filename, std::ios::binary);
    if (!ofile.is_open()){std::cerr<<"could not open output file"<<std::endl;exit(0);}

    char _desc_name[20];
    desc_name.resize(std::min(size_t(19),desc_name.size()));
    strcpy(_desc_name,desc_name.c_str());
    ofile.write(_desc_name,20);

    uint32_t size=features.size();
    ofile.write((char*)&size,sizeof(size));
    for(auto &f:features){
        if( !f.isContinuous()){
            std::cerr<<"Matrices should be continuous"<<std::endl;exit(0);
        }
        uint32_t aux=f.cols; ofile.write( (char*)&aux,sizeof(aux));
        aux=f.rows; ofile.write( (char*)&aux,sizeof(aux));
        aux=f.type(); ofile.write( (char*)&aux,sizeof(aux));
        ofile.write( (char*)f.ptr<uchar>(0),f.total()*f.elemSize());
    }
}


// ----------------------------------------------------------------------------
std::vector<cv::Mat> readFeaturesFromFile(std::string filename,std::string &desc_name) {
    std::vector<cv::Mat> features;
    //test it is not created
    std::ifstream ifile(filename,std::ios::binary);
    if (!ifile.is_open())
    {
        std::cerr<<"could not open input file"<<std::endl;
        exit(0);
    }


    char _desc_name[20];
    ifile.read(_desc_name,20);
    desc_name=_desc_name;

    uint32_t size;
    ifile.read((char*)&size,sizeof(size));
    features.resize(size);
    for(size_t i=0;i<size;i++){

        uint32_t cols,rows,type;
        ifile.read( (char*)&cols,sizeof(cols));
        ifile.read( (char*)&rows,sizeof(rows));
        ifile.read( (char*)&type,sizeof(type));
        features[i].create(rows,cols,type);
        ifile.read( (char*)features[i].ptr<uchar>(0),features[i].total()*features[i].elemSize());
    }
    return features;
}

//double computeCosine

int main(int argc,char **argv) {
    try{
            CmdLineParser cml(argc,argv);
            if (cml["-h"] || argc<4){
                std::cerr<<"Usage:  descriptor_name train_dir_with_images  test_dir_with_images\n\t descriptors:brisk,surf,orb(default),akaze(only if using opencv 3)"<<std::endl;
                return -1;
            }

            std::string descriptor=argv[1];
            std::string desc_file= descriptor+".txt";
            std::string fbow_file = descriptor+"_fbow.fbow";

            auto train_images = DirReader::read(argv[2]);
            std::sort(train_images.begin(), train_images.end());
            
            std::vector< cv::Mat   >   train_features= loadFeatures(train_images,descriptor);

            //save features to file
            std::cerr<<"saving to "<<desc_file<<std::endl;
            saveToFile(desc_file, train_features, descriptor);

            auto train_features_loaded = readFeaturesFromFile(desc_file, descriptor);
            std::cout<<"DescName="<<descriptor<<std::endl;
            fbow::VocabularyCreator::Params params;
            params.k = stoi(cml("-k","10"));
            params.L = stoi(cml("-l","6"));
            params.nthreads=stoi(cml("-t","1"));
            params.maxIters=std::stoi (cml("-maxIters","0"));
            params.verbose=cml["-v"];
            srand(0);
            fbow::VocabularyCreator voc_creator;
            fbow::Vocabulary voc;
            std::cout << "Creating a " << params.k << "^" << params.L << " vocabulary..." << std::endl;
            auto t_start=std::chrono::high_resolution_clock::now();
            voc_creator.create(voc,train_features_loaded, descriptor, params);
            auto t_end=std::chrono::high_resolution_clock::now();
            std::cout<<"time="<<double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count())<<" msecs"<<std::endl;
            std::cout<<"nblocks="<<voc.size()<<std::endl;
            
            std::cerr<<"Saving "<<fbow_file<<std::endl;
            voc.saveToFile(fbow_file);

            // load fbows vocabulary
            voc.readFromFile(fbow_file);
            std::cout<< "========================= Query search part =========================" << std::endl;
            // query images
            auto test_images= DirReader::read( argv[3]);
            std::sort(test_images.begin(), test_images.end());

            for (size_t i = 2; i < test_images.size(); ++i)
                std::cout << test_images[i] << std::endl;
            std::vector< cv::Mat   >   test_features= loadFeatures(test_images, descriptor);
            //std::vector<fbow::fBow> bowVectors;
            for (int i = 0; i < test_features.size(); ++i) {
                auto &f = test_features[i];
                fbow::fBow bowVector1 = voc.transform(f);
                
                std::map<double, int> score;
                for (size_t j = 0; j<train_features_loaded.size(); ++j)
                {

                    fbow::fBow  bowVector2 = voc.transform(train_features_loaded[j]);
                    double score1 = bowVector1.score(bowVector1, bowVector2);
                    //counter++;
                    //		if(score1 > 0.01f)
                    {
                        score.insert(std::pair<double, int>(score1, j));
                    }
                }

                std::vector<std::pair<double, int>> sorted_score(score.begin(), score.end());
                std::sort(sorted_score.begin(), sorted_score.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                    return a.first > b.first;
                });
                for (const auto& s : sorted_score) {
                    std::cout << "Matching score: " << s.first << ", Index: " << s.second << std::endl;
                }

                std::cout << "=============== Displaying most similar image to query image ==================" << std::endl;
                // viewing similar images
                std::vector<cv::Mat> result_images;
                cv::Mat img1 = cv::imread(test_images[i+2]);
                cv::Mat img2 = cv::imread(train_images[sorted_score[0].second+2]);
                std::cout << "Query image_size: " << img1.size() << std::endl;
                std::cout << "Most similar image_size: " << img2.size() << std::endl;
                result_images.push_back(img1);
                result_images.push_back(img2);

                cv::Mat combined;
                cv::hconcat(result_images, combined);
                cv::imshow("Query and Top Matches", combined);
                cv::waitKey(0);
            }
            

        } catch(std::exception &ex) {
            std::cerr<<ex.what()<<std::endl;
        }

        return 0;
}