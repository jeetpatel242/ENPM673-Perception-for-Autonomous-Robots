# ENPM673 Perception for Autonomous Robots
## Project-2 Lane detection and Turn Prediction

## Overview
This project aims to do Lane Detection to mimic Lane Departure Warning systems used in Self Driving Cars. Datasets are provided with two video sequences, taken from a self-driving car. Task is to detect lanes on the road, as well as predict car turns.


## Dependencies
* Python3
* OpenCV
* os
* Numpy

## Install Dependencies
* Install Python3, and the necessary libraries: (if not already installed)
````
sudo apt install python3
sudo apt install python3-opencv
pip install numpy
````

### Run
* Unzip the folder into your system
````
Copy the following required dataset folders/files into the above directory:
-- Night Drive - 2689.mp4
-- data_1 (all images should be in data folder as provided)
-- data_2 (It should cotain challenge_video.mp4)
````
* Open terminal 
* Double check if all dependencies are installed
* run solution of problem 1.
* Program will generate the output video file of enhanced video.
````
python3 Problem-1.py
````
* run solution of Problem-2
* It will ask the user to choose the dataset (1 or 2).
* Program will generate the output video file as well as visulization will start.
````
python3 Problem-2.py
````
Output video files will be saved in same directory.

## Maintainer
* Jeet Patel (jeetpatel242@gmail.com)