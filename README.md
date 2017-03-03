Background Subtraction with Dual Single Gaussian Model
================================================================================

This is my implementation of the algorithm described in the paper "Detection of 
Moving Objects with Non-stationary Cameras in 5.8ms: Bringing Motion Detection to 
Your Mobile Device," Yi et al, CVPRW 2013. Currently, it only works well on 
sequences from stationary camera.


System Dependency
--------------------------------------------------------------------------------

* Opencv
* CMake


Compilation and usage
--------------------------------------------------------------------------------

* In command line, go to project root directory

  1. mkdir build
  2. cd build
  3. cmake ..
  4. make

* Once the project is built, you can try:

  '''bash
  ./dual_sgm ../data/walking.avi
  '''
