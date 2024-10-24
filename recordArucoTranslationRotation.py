# Joel Currie

import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import os
from scipy.spatial.transform import Rotation   
import time
import pandas as pd

def write2csv(id, t_out, marker_id, x_out, y_out, z_out, theta_out):
    out_mat = np.array([t_out, marker_id, x_out, y_out, z_out, theta_out], dtype = np.float32)
    out_mat = out_mat.transpose()
    df = pd.DataFrame(out_mat)
    df.columns = ['t_out', 'marker_id', 'x_out', 'y_out', 'z_out', 'theta_out']
    df = df[df['marker_id'] == 2]
    path = "data-" + str(id) + ".csv"
    df.to_csv(path, header = ['t_out', 'marker_id', 'x_out', 'y_out', 'z_out', 'theta_out'])

def getCameraParams(name = 'calibration_chessboard.yaml'):
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    os.chdir(current_directory)
    camera_calibration_parameters_filename = name
    
    # Load the camera parameters from the saved file
    cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()
    return mtx, dst

def localiseAruco(t_out, x_out, y_out, z_out):
    video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    #Error checking
    if not video.isOpened():
        print ("ERROR: COULD NOT OPEN VIDEO ")
        sys.exit()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    while(True):
        #read frame
        ok, frame = video.read()
        if not ok:
            print ("ERROR: COULD NOT READ FIRST FRAME FROM FILE")
            sys.exit()
        #frame operations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        ids_filt = np.ndarray(shape=(1,1), dtype=float)
        corners_filt = np.ndarray(shape=(1, 1, 4, 2), dtype=np.float32)
        if type(ids) is not type(None):
            for i in range(len(ids)):
                if ids[i] == 2:
                    ids_filt[0] = 2
                    corners_filt[0] = corners[i]
        frame = aruco.drawDetectedMarkers(frame, corners_filt, ids_filt, borderColor=(0,255,0))   
        try:
            rvec, tvec, obj_points = (aruco.estimatePoseSingleMarkers(
                corners_filt, 
                0.03, 
                mtx, 
                dst))
            rot_mat, _ = cv2.Rodrigues(np.array(rvec))
            r =  Rotation.from_matrix(rot_mat)
            angles = r.as_euler("zyx",degrees=True)
            theta = angles[0]
            
            t = time.time() - t0
            t_x = np.array(tvec[0][0][0])
            t_y = np.array(tvec[0][0][1])
            t_z = np.array(tvec[0][0][2])


            t_out.append(t)
            x_out.append(t_x)
            y_out.append(t_y)
            z_out.append(t_z)
            theta_out.append(theta)
            marker_id.append(ids_filt[0][0])

        except Exception as error:
            print("An exception occurred:", error)


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return t_out, x_out, y_out, z_out

def localiseArucoRealsense(t_out, marker_id, x_out, y_out, z_out, theta_out):
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
    # Load the camera parameters from the saved file
    cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()


    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    try:
        while True:
    
            
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
    
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            frame = color_image
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            
            scaled_depth=cv2.convertScaleAbs(depth_image, alpha=0.08)
            depth_colormap = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)
    
            # Stack both images horizontally

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            frame = aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0,255,0))
            images = np.hstack((frame, depth_colormap))

            cv2.imshow('RealSense', images)
            try:
                rvec, tvec, obj_points = (aruco.estimatePoseSingleMarkers(
                corners_filt, 
                0.03, 
                mtx, 
                dst))
                rot_mat, _ = cv2.Rodrigues(np.array(rvec))
                r =  Rotation.from_matrix(rot_mat)
                angles = r.as_euler("zyx",degrees=True)
                theta = angles[0]
            
                t = time.time() - t0
                t_x = np.array(tvec[0][0][0])
                t_y = np.array(tvec[0][0][1])
                t_z = np.array(tvec[0][0][2])


                t_out.append(t)
                x_out.append(t_x)
                y_out.append(t_y)
                z_out.append(t_z)
                theta_out.append(theta)
                marker_id.append(ids_filt[0][0])

            except Exception as error:
                print("An exception occurred:", error)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
    
    finally:
    
        # Stop streaming
        pipeline.stop()

def getParticipantID():
    print("Enter Participant ID:")
    id = input()
    return id



t0 = time.time()
verbose_out = True

t_out = []
x_out = []
y_out = []
z_out = []
theta_out = []
marker_id = []


id = getParticipantID()
mtx, dst = getCameraParams(name = 'calibration_chessboard.yaml')
t_out, x_out, y_out, z_out = localiseAruco(t_out, x_out, y_out, z_out)
#t_out, x_out, y_out, z_out = localiseArucoRealsense(t_out, x_out, y_out, z_out)
write2csv(id, t_out, marker_id, x_out, y_out, z_out, theta_out)