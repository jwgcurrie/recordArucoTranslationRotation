## recordArucoTranslationRotation
Outputs translation and rotation in euler angles of specified Aruco marker as .csv using opencv.

# 1. Setup Python Environment:
Using conda (https://anaconda.org/anaconda/conda) create a new virtual environment using the file .Tracker.yml. Use the following command to do so.

$conda env create --name ePuckTracker --file=ePuckTracker.yml

# 2. Print all files in the 'materials' directory

# 3. Calibrate Camera:
Take a mininum of 10 photos from your camera in .jpg format of the chessboard at different locations and different rotations.
Add these to the 'calibrate' folder and run calibration.py. This will calculate the distortion coefficients of your camera.

# 4. Run 'recordArucoTranslationRotation':
Run the script 'recordArucoTranslationRotation', if you have a regular webcam, use the funciton localiseAruco(), if you are using an intelRealsense camera, use localiseArucoRealsense().

First the program will ask you to enter an ID, this is simply a way of identifying the .csv file that will be generated. 
A window will open showing the realtime detection of the aruco marker.
Important, if you want the localisation data to be recorded in a .csv file, press 'q' to exit the program. 

