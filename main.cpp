#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#include <cstdio>
#include <cstdint>

//opencv headers
#include <random>
#include <opencv2/core/mat.hpp>   //cv::Mat
#include <opencv2/highgui.hpp>    //cv::imshow, cv::waitKey
#include <opencv2/imgproc.hpp>  //cv::imread

#include "timer.h"


static const unsigned int COLS = 1920 ;
static const unsigned int ROWS = 1080 ;

static const unsigned int DEAD = 0;
static const unsigned int ALIVE = 255;

static const unsigned int GENERATIONS = 1000;

int main() {
  cv::Mat buffer1(ROWS, COLS, CV_8UC1);
  cv::Mat buffer2(ROWS, COLS, CV_8UC1);
  
  std::mt19937_64 engine(
      static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
  std::uniform_real_distribution<double> rngDistribution(0.0, 1.0);
  
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col < COLS; col++) {
      if (row == 0 || col == 0 || row == ROWS - 1 || col == COLS - 1) {
        buffer1.at<uint8_t>(row, col) = DEAD;
      } else {
        if (rngDistribution(engine) > 0.9) {
          buffer1.at<uint8_t>(row, col) = ALIVE;
        } else {
          buffer1.at<uint8_t>(row, col) = DEAD;
        }
      }
    }
  }
  cv::Mat *renderBufferPtr = &buffer2;
  cv::Mat *modelBufferPtr = &buffer1;
  
  uint64_t generation = 0;
  cv::namedWindow("Game of life", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
  cv::setWindowProperty("Game of life", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  
  Timer logicTimer;
  float logicMs = 0.f;
  
  Timer renderTimer;
  float renderMs = 0.f;
  
  
  float sumTime = logicMs;
  cv::Mat rgbDebugRenderImg(ROWS, COLS, CV_8UC3);
  do {
    //rendering
    renderTimer.reset();
    {
      if (renderBufferPtr == &buffer1) {
        renderBufferPtr = &buffer2;
        modelBufferPtr = &buffer1;
      } else {
        renderBufferPtr = &buffer1;
        modelBufferPtr = &buffer2;
      }
      
      #pragma omp parallel for num_threads(4)
      for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
          uint8_t pix = renderBufferPtr->at<uint8_t>(row, col);
          rgbDebugRenderImg.at<cv::Vec3b>(row, col) = cv::Vec3b(pix, pix, pix);
        }
      }
      cv::putText(rgbDebugRenderImg, //target image
                  "Logic: " + std::to_string(logicMs) + " ms", //text
                  cv::Point(10, 30), //top-left position
                  cv::FONT_HERSHEY_DUPLEX,
                  1.f,
                  CV_RGB(255, 0, 0), //font color
                  2);
      
      
      cv::putText(rgbDebugRenderImg, //target image
                  "Render: " + std::to_string(renderMs) + " ms", //text
                  cv::Point(10, 60), //top-left position
                  cv::FONT_HERSHEY_DUPLEX,
                  1.f,
                  CV_RGB(255, 0, 0), //font color
                  2);
      
      cv::putText(rgbDebugRenderImg, //target image
                  "Generation: " + std::to_string(generation), //text
                  cv::Point(10, 90), //top-left position
                  cv::FONT_HERSHEY_DUPLEX,
                  1.f,
                  CV_RGB(255, 0, 0), //font color
                  2);
//    cv::imshow("Game of life", *renderBufferPtr);
      cv::imshow("Game of life", rgbDebugRenderImg);
      *modelBufferPtr = renderBufferPtr->clone();
    }
    renderMs = renderTimer.elapsed();
    
    //Game logic
    logicTimer.reset();
    {
      #pragma omp parallel for num_threads(4)
      for (int row = 1; row < ROWS - 1; row++) {
        for (int col = 1; col < COLS - 1; col++) {
          uint8_t pix = modelBufferPtr->at<uint8_t>(row, col);
          
          int aliveNeighbors = 0;
          aliveNeighbors = renderBufferPtr->at<uint8_t>(row - 1, col - 1) +
                           renderBufferPtr->at<uint8_t>(row - 1, col) +
                           renderBufferPtr->at<uint8_t>(row - 1, col + 1) +
                           renderBufferPtr->at<uint8_t>(row, col - 1) +
                           renderBufferPtr->at<uint8_t>(row, col + 1) +
                           renderBufferPtr->at<uint8_t>(row + 1, col - 1) +
                           renderBufferPtr->at<uint8_t>(row + 1, col) +
                           renderBufferPtr->at<uint8_t>(row + 1, col + 1);
          aliveNeighbors /= ALIVE;
          
          if (pix == DEAD) {
            if (aliveNeighbors == 3) {
              //Jesus f***ing Christ is alive
              pix = ALIVE;
            }
          } else {
            if (aliveNeighbors < 2 || aliveNeighbors > 3) {
              //Die motherfucker
              pix = DEAD;
            }
            
            if (aliveNeighbors == 2 || aliveNeighbors == 3) {
              //Stay alive... for now
              pix = ALIVE;
            }
          }
          modelBufferPtr->at<uint8_t>(row, col) = pix;
        }
      }
    }
    
    logicMs = logicTimer.elapsed() * 1000.f;
    printf("Generation: %d\t elapsed: %fms\t render: %fms\n", generation++, logicMs, renderMs);
    sumTime += logicMs;
  } while (cv::waitKey(1) < 0 /*&& generation < GENERATIONS*/);
  
  float avgTime = sumTime / generation;
  
  printf("Average time: %f ms\n", avgTime);
  
  return 0;
}

#pragma clang diagnostic pop