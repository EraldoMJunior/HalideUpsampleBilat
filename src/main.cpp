#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>


// The only Halide header file you need is Halide.h. It includes all of Halide.
#include "Halide.h"
#include "HalideBuffer.h"


// Include some support code for loading pngs.
#include "halide_image_io.h"
#include "bilateral_upsample.h"
#include <chrono>



using namespace Halide::Tools;
using namespace Halide;

cv::Mat create_gaussian_filter(int sigma) {

    int rmax = 2.5 * sigma;
    int window_size = 2 * rmax + 1;

    cv::Mat Gaussian_filter = cv::Mat::zeros(window_size, window_size, CV_32F);

    for (int i = 0; i < window_size; ++i) {
        for (int j = 0; j < window_size; ++j) {
            float value = std::exp(-(((i - rmax) * (i - rmax) + (j - rmax) * (j - rmax)) / (2.0 * sigma * sigma)));
            Gaussian_filter.at<float>(i, j) = value;
        }
    }
    return Gaussian_filter;
}

// cv::Mat bilateral_filter(cv::Mat image, int sigma_s, int sigma_r) {

//     cv::Mat Gaussian_filter = create_gaussian_filter(sigma_s);

//     int window_size = Gaussian_filter.size().height;
//     int rmax = window_size / 2;

//     int height = image.size().height;
//     int width = image.size().width;

//     cv::Mat blurred = cv::Mat::zeros(height, width, CV_8UC1);

//     for (int i = 0; i < height; ++i) {
//         for (int j = 0; j < width; ++j) {

//             float sum = 0;
//             float normalize = 0;

//             for (int dx = -rmax; dx <= rmax; ++dx) {
//                 for (int dy = -rmax; dy <= rmax ; ++dy){

//                     //avoid out of boundary exception
//                     if (i + dx < 0 || i + dx >= height || j + dy < 0 || j + dy >= width) continue;

//                     //calculate range difference
//                     int intensity_difference = (image.at<uchar>(i, j) - image.at<uchar>(i + dx, j + dy)) * (image.at<uchar>(i, j) - image.at<uchar>(i + dx, j + dy));

//                     //final weight, combination of the range kernel and the Gaussian kernel
//                     float weight = std::exp(-((intensity_difference) / (2.0 * sigma_r * sigma_r))) * Gaussian_filter.at<float>(dx +rmax , dy+rmax);

//                     //accumulate the weights for normalization
//                     normalize += weight;

//                     //taking the weighted sum of the input values
//                     sum += weight * image.at<uchar>(i + dx, j + dy);
//                 }
//             }

//             blurred.at<uchar>(i, j) = std::round(sum / normalize);
//         }
//     }

//     return blurred;

// }








 //int bilateral_upsample(struct halide_buffer_t *_rgb_buffer, struct halide_buffer_t *_mask_buffer, struct halide_buffer_t *_output_buffer);
void  upsample_halide_run(const char* filename  , const char* secundary  )
{
    int sig_s = 4;
    Halide::Runtime::Buffer<float> k(2*sig_s + 1, 2*sig_s + 1 );
    k.translate({-sig_s, -sig_s});
    k.fill(0.f);
    for (int i = -sig_s; i <= sig_s; ++i) {
        for (int j = -sig_s; j <= sig_s; ++j) {
            float value = std::exp(-(((i ) * (i  ) + (j  ) * (j  )) / (2.0 * sig_s * sig_s)));
            k(i,  j) = value;
        }
    }

    Halide::Runtime::Buffer<uint8_t> input = load_image(filename);
    Halide::Runtime::Buffer<uint8_t> g_sec = load_image(secundary);
    int width = input.width();
    int height = input.height();
    Halide::Runtime::Buffer<uint8_t> output(width, height );

    for(int loop = 0 ; loop < 3 ; loop++){
       auto t_start  = std::chrono::high_resolution_clock::now();
       bilateral_upsample( input , g_sec , k, output   );
         auto t_end  = std::chrono::high_resolution_clock::now();
            std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms" << std::endl;
    }

    save_image(output, "output.png");

    return ;
}




cv::Mat joint_bilateral_filter(cv::Mat f, cv::Mat g, int sigma_s, int sigma_r) {

    cv::Mat Gaussian_filter = create_gaussian_filter(sigma_s);
    int window_size = Gaussian_filter.size().height;
    int rmax = window_size / 2;

    int height = f.size().height;
    int width = f.size().width;

    cv::Mat blurred = cv::Mat::zeros(height, width, CV_8UC1);

    #pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            int xmin = std::max( i - 2 , 0);
            int xmax = std::min( i + 2 , height-1);
            int ymin = std::max( j - 2 , 0);
            int ymax = std::min( j + 2 , width-1);

            uchar p0  =  g.at<uchar>( i , j);
            uchar p1  =  g.at<uchar>( xmin , j);
            uchar p2  =  g.at<uchar>( xmax , j);
            uchar p3  =  g.at<uchar>( i , ymin);
            uchar p4  =  g.at<uchar>( i , ymax);


     if (p0 <= 10 || p0 >= 255 - 10 ){
          blurred.at<uchar>(i, j) = p0;
                        continue;
          }

            float sum = 0;
            float normalize = 0;

            for (int dx = -rmax; dx <= rmax; ++dx) {
                for (int dy = -rmax; dy <= rmax ; ++dy) {
                    if (i + dx < 0 || i + dx >= height || j + dy < 0 || j + dy >= width) continue;
                    int intensity_difference = (f.at<uchar>(i, j) - f.at<uchar>(i + dx, j + dy)) * (f.at<uchar>(i, j) - f.at<uchar>(i + dx, j + dy));
                    float weight = std::exp(-((intensity_difference) / (2.0 * sigma_r * sigma_r))) * Gaussian_filter.at<float>(dx+rmax, dy+rmax);
                    normalize += weight;
                    sum += weight * g.at<uchar>(i + dx, j + dy);
                }
            }
            blurred.at<uchar>(i, j) = std::round(sum / normalize);
        }
    }

    return blurred;
}

cv::Mat upsample(int sigma_s, int sigma_r, cv::Mat disp, cv::Mat image) {

    int upsample_factor = std::floor(std::log2(image.size().height / (float)disp.size().height)); //  round down to the smaller int number

    cv::Mat upsampled_disp = disp;

    for (int i = 1; i < upsample_factor - 1; ++i) {
        std::cout << "Upsampling disparity image... " << std::floor(((float)i / upsample_factor) * 100) << "%\r" << std::flush;
        cv::resize(upsampled_disp, upsampled_disp, cv::Size(), 2.0, 2.0);
        cv::Mat lowres_image;
        cv::resize(image, lowres_image, upsampled_disp.size());
        upsampled_disp = joint_bilateral_filter(lowres_image, upsampled_disp, sigma_s, sigma_r);
    }
    std::cout << "Upsampling disparity image... " << std::floor(((float)(upsample_factor - 1) / upsample_factor) * 100) << "%\r" << std::flush;
    cv::resize(upsampled_disp, upsampled_disp, image.size());
    upsampled_disp = joint_bilateral_filter(image, upsampled_disp, sigma_s, sigma_r);
    std::cout << "Upsampling disparity image... Done." << std::endl;
    return upsampled_disp;
}



int main(int argc, char** argv) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << "  RGB_image  Depth_image Sigma_S Sigma_R" << std::endl;
        return 1;
    }

    int sigma_s = std::stoi(argv[3]);
    int sigma_r = std::stoi(argv[4]);

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    std::cout << "Calculating filtered images... Done.     " << std::endl;

    cv::Mat disp = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    //check if disp is 8x smaller than image
    if (disp.size().height * 8 != image.size().height || disp.size().width * 8 != image.size().width) {
//dump sizes
       std::cout << "image size " << image.size().height << " " << image.size().width << std::endl;
         std::cout << "disp size " << disp.size().height << " " << disp.size().width << std::endl;

         std::cout<< "Please resize to " << image.size().height / 8 << " " << image.size().width / 8 << std::endl;

        std::cerr << "Error: Depth image must be 8x smaller than RGB image." << std::endl;

        return 1;
    }

    upsample_halide_run(  argv[1] ,  argv[2] );


    cv::Mat upsampled_disp = upsample(sigma_s, sigma_r, disp, image);

    // thresholding OSTU

     //cv::threshold(upsampled_disp, upsampled_disp, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
     //apply sigmoid
        for(int xi = 0 ; xi < upsampled_disp.cols ; xi++)
            for(int yi = 0 ; yi < upsampled_disp.rows ; yi++){
                // sigmoid using 0 -255 range
                u_char p = upsampled_disp.at<u_char>(yi,xi);
                float px = 15*(p / 255.0 - 0.5f);
                float sig = 1 / (1 + exp(-px));
                upsampled_disp.at<u_char>(yi,xi) = sig * 255;
            }




    cv::imwrite("upsampled_disp.png", upsampled_disp);



    return 0;
}
