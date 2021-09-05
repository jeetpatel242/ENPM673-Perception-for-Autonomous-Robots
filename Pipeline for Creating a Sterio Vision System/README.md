# ENPM673 Perception for Autonomous Robots
## Project-3 Creating a Stereo Vision System

## Overview
This project aims to implement the concept of Stereo Vision. 3 different datasets are given, each of them contains 2 images of the same scenario but taken from two different camera angles. By comparing the
information about a scene from 2 vantage points, we can obtain the 3D
information by examining the relative positions of objects.

## Dependencies
* Python3
* OpenCV
* matplotlib
* Numpy

## Install Dependencies
* Install Python3, and the necessary libraries: (if not already installed)
````
sudo apt install python3
sudo apt install python3-opencv
python -m pip install -U matplotlib
pip install numpy
````

### Run
* Unzip the folder into your system
````
Copy the following required dataset folders/files into the above directory:
-- Dataset 1
-- Dataset 2
-- Dataset 3
````
* Open terminal 
* Double check if all dependencies are installed
* Run the following command in terminal 
````
python3 code.py
````
* The program will ask the dataset. Enter (1 or 2 or 3).
* The program will further ask for function to be carried out (1 = Sterio Rectification or 2 = disparity and depth map).
* Choosing stereo rectification will generate epipolar lines along with feature points, while second function will generate heat map of disparity and depth image.

## Maintainer
* Jeet Patel (jeetpatel242@gmail.com)