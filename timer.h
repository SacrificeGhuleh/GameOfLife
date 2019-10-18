//
// Created by zvone on 03-Oct-19.
//

#ifndef PA1_2019_TIMER_H
#define PA1_2019_TIMER_H

#include <iostream>
#include <chrono>
#include <omp.h>

class Timer {
public:
  Timer() : beg_(clock_::now()) {}
  
  void reset() { beg_ = clock_::now(); }
  
  double elapsed() const {
    return std::chrono::duration_cast<second_>
        (clock_::now() - beg_).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point<clock_> beg_;
};


#endif //PA1_2019_TIMER_H
