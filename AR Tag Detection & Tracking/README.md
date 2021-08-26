# ENPM673 Perception for Autonomous Robots
## Project 1 AR Tag Detection and Tracking

## Overview
This project detects and tracks the AR tags and implement augmented reality applications in the video.


## Dependencies
* Python3
* OpenCV
* Matplotlib
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
Copy the following required dataset files into the above directory:
-- ref_marker.png
-- testudo.png
-- Tag0.mp4
-- Tag1.mp4
-- Tag2.mp4
-- multipleTags.mp4

````
* Open terminal 
* Double check if all dependencies are installed
* run solution of problem 1A.
* Progam will plot varoius graph and will detect contour for a single frame.
````
python3 Problem-1a.py
````
* run solution of Problem-1B
* Progam will plot the curve of grid of tag and also gives Tag ID and orientation.
````
python3 Problem-1b.py
````
* run solution of problem-2
* Enter the input of video name without extension.
````
For example, Tag0.mp4: Tag0
````
* Select 1 for imposing the image of testudo and 2 for imposing a 3D cube.
* The code will display the each frame process and also generates the output video file.
````
python3 problem-2.py
````


## Maintainer
* Jeet Patel (jeetpatel242@gmail.com)