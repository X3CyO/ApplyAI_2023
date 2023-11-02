
import cv2 as cv
import time
import utils, math
import numpy as np
import csv
import mediapipe as mp
import torch
import math
import shutil #for duplicating files
import os

# Initialize the face landmarks detection module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Download the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Load depth transformation functions
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth
    return filtered_depth

def depth_to_distance(depth_value, depth_scale):
    return 1.0 / (depth_value * depth_scale)

# Open the camera you want; can run camera_select.py to find which camera is which by integer/parameters.
camera = cv.VideoCapture(0)  # Replace # with the appropriate camera index
# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Camera not found.")
else:
    # Set the desired video quality (resolution) aspect ratio = 16:9
    # 1080p = 1920x1080 (Full HD) -> 720p = 1280x720 (Standard)  -> 480p = 640x480 (Low)
    # 14 is around the highest fps I can get at 640x480? (lowest I can go) (only need >1/3 of a second; 40fps.. or .3 sec)
    
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv.CAP_PROP_FPS, 60)  # Set frame rate

    # Check if parameters were set successfully
    print(f"Video width: {camera.get(cv.CAP_PROP_FRAME_WIDTH)}")
    print(f"Video height: {camera.get(cv.CAP_PROP_FRAME_HEIGHT)}")
    print(f"Frame rate: {camera.get(cv.CAP_PROP_FPS)}")

# Create and open a CSV file for logging
csv_filename = "current_eye_status_log.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# variables 
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0

# constants
CLOSED_EYES_FRAME = 0
FONTS = cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

# left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def blinkRatioWithStabilityAndTracking(img, landmarks, right_indices, left_indices, prev_frame, prev_landmarks=None, stability_threshold=0.1):
    if prev_landmarks is not None:
        # Calculate optical flow to track landmarks from the previous frame to the current frame
        updated_landmarks, _ = optical_flow_tracking(prev_frame, img, prev_landmarks)

        # Calculate motion between tracked landmarks
        motion = calculate_motion(landmarks, updated_landmarks)

        # Check stability based on motion magnitude
        if motion > stability_threshold:
            return None  # Frame is considered unstable
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eye
    cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    # draw lines on left eye
    cv.line(img, lh_left, lh_right, utils.GREEN, 2)
    cv.line(img, lv_top, lv_bottom, utils.WHITE, 2)

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio , updated_landmarks

def optical_flow_tracking(prev_frame, current_frame, prev_landmarks):
    # Convert the list of landmarks to a NumPy array
    prev_points = np.array(prev_landmarks, dtype=np.float32)

    # Calculate optical flow
    current_points, status, _ = cv.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None)

    # Filter out points with low tracking quality
    good_points = prev_points[status == 1]
    updated_points = current_points[status == 1]

    # Convert the updated points back to a list
    updated_landmarks = updated_points.tolist()

    return updated_landmarks

def calculate_motion(current_landmarks, prev_landmarks):
    # Calculate motion between previous and current landmarks
    motion = sum(euclaideanDistance(curr, prev) for curr, prev in zip(current_landmarks, prev_landmarks))
    return motion

min_detection_confidence = .5 #originally .5; set to 0 to count when the person isnt looking as a ratio to be determined. <1?
min_tracking_confidence = .8 #originally .5
last_timestamp = None
                # This determines how open/closed the eye is to append a blink... 4 works well; just might be too much... what is the value of lost blinks vs fakes..?
                # this also highly depends on how far you are... the further away, the more sensitive the ratio since the eye is smaller.
#ratio_criteria = 4 # at this # a blink is counted; which has been dynamically accounted for now in the latter code

with map_face_mesh.FaceMesh(min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence) as face_mesh:
    # starting time here 
    start_time = time.time()
    
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        face_results = face_mesh.process(frame)

        # Calculate the elapsed time and format it as HH:MM:SS:#/60
        elapsed_time = time.time() - start_time


        # Calculate the elapsed time in seconds and fractions
        elapsed_seconds = int(elapsed_time)
        fraction = int((elapsed_time - elapsed_seconds) * 60)

        # Format the elapsed time as H:M:S
        elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))

        if face_results.multi_face_landmarks is not None:
            for face_landmarks in face_results.multi_face_landmarks:
                # Calculate the distance between point 10 and point 152 (top and bottom of face)
                landmark_10 = face_landmarks.landmark[10]
                landmark_152 = face_landmarks.landmark[152]

                # Calculate the 3D distance
                distance1 = math.sqrt((landmark_10.x - landmark_152.x)**2 + (landmark_10.y - landmark_152.y)**2 + (landmark_10.z - landmark_152.z)**2)

                # Calculate the distance between point 234 and 454 (left and right side of face)
                landmark_234 = face_landmarks.landmark[234]
                landmark_454 = face_landmarks.landmark[454]

                # Calculate the 3D distance
                distance2 = math.sqrt((landmark_234.x - landmark_454.x)**2 + (landmark_234.y - landmark_454.y)**2 + (landmark_234.z - landmark_454.z)**2)

                # Calculate the average of both distances
                average_distance = (distance1 + distance2) / 2

                # Adjust the mid-point as needed
                mid_x, mid_y, mid_z = 0, 0, 0

                # Define the new position for the text (upper right corner)
                x = frame.shape[1] - 1000  # Adjust this value to control the horizontal position
                y = 75  # Adjust this value to control the vertical position
                
                # Black outline
                cv.putText(frame, "Relative distance ratio from center of face: " + str(round(average_distance, 2)), (x, y), FONTS,
                        1,(0,0,0), 5)
                
                # White fill
                cv.putText(frame, "Relative distance ratio from center of face: " + str(round(average_distance, 2)), (x, y), FONTS,
                        1, (255, 255, 255), 3)
                
            #  resizing frame
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)

            # Convert elapsed_time to a human-readable format (e.g., HH:MM:SS)
            elapsed_time_sec = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

            # Display elapsed time on the frame
            cv.putText(frame, f'Elapsed Time: {elapsed_time_sec}', (10, 30), FONTS, 1, (0, 255, 0), 2) 

            # Set the degree change per 0.01 point change in ratio
            x = average_distance
            ratio_criteria = (-6.78*x + 6.14)  # Adjust this value as needed

            # Determine if a person is in view based on face detection confidence
            if results.multi_face_landmarks:
                    person_in_view = True #pseudo code
                    mesh_coords = landmarksDetection(frame, results, False)
                    ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

                    # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                    utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                    utils.colorBackgroundText(frame,  f'Distance Weight : {round(ratio_criteria,2)}', FONTS, 0.7, (30,200),2, utils.PINK, utils.YELLOW)
                    # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                    utils.colorBackgroundText(frame,  f'Total Frames w/ Eyes Closed: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    
                    if ratio >= ratio_criteria:
                        CEF_COUNTER += 1
                        # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                        utils.colorBackgroundText(frame,  f'Eyes Closed', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6)

                    if ratio < ratio_criteria:
                        # Log the frame index and eye status (1 for open)
                        csv_writer.writerow([elapsed_time_str, fraction, frame_counter, 1])
                        utils.colorBackgroundText(frame,  f'Eyes Open', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6)

                    else:
                        if CEF_COUNTER > CLOSED_EYES_FRAME:
                            TOTAL_BLINKS += 1
                            CEF_COUNTER = 0

                            # Log the frame index and eye status (0 for closed)
                            csv_writer.writerow([elapsed_time_str, fraction, frame_counter, 0])
                            # No frame, log a blank entry
        else:
            person_in_view = False #pseudo code
            if not person_in_view:
                csv_writer.writerow([elapsed_time_str, fraction, frame_counter, -1])
                utils.colorBackgroundText(frame,  f'Out of Frame', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6)
        if not ret: 
            break # no more frames break    

        # Calculate FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time
        frame = utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv.imshow('frame', frame)
        cv.imshow('Face Distance', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
        
    cv.destroyAllWindows()
    camera.release()

    # Calculate FPS
    end_time = time.time() - start_time
    fps = frame_counter / end_time
    print(f"Frames per second (FPS): {round(fps, 1)}")

    # Close the CSV file at the end of the script
    csv_file.close()

# Folder paths
original_folder = "original_data"

# Create the folders if they don't exist
os.makedirs(original_folder, exist_ok=True)

# Make a copy of the original CSV file
timestamp = time.strftime("%Y%m%d%H%M%S")
duplicated_csv_filename = f"2original_eye_status_log_duplicate_{timestamp}.csv"
shutil.copy(csv_filename, os.path.join(original_folder, duplicated_csv_filename))
print(f"Original CSV file duplicated with timestamp: {duplicated_csv_filename}")


# Interpolate the missing 1/60 segments

output_csv_filename = "hz_interpolated_eye_status_log.csv"

import csv

def interpolate_data(csv_filename, output_csv_filename):
    # Initialize the CSV reader and writer
    with open(csv_filename, 'r') as csv_in, open(output_csv_filename, 'w', newline='') as csv_out:
        csv_reader = csv.reader(csv_in)
        csv_writer = csv.writer(csv_out)

        last_time, last_value = 0, 0

        for row in csv_reader:
            timestamp = row[0]  # Extract the timestamp from the first column
            values = [int(value) if value != '' else '' for value in row[1:]]  # Extract values from the second column onward

            # Check if the fractional seconds have rolled over to 0
            if values[1] == 0 and last_value == 59:
                values[0] = (values[0] + 1) % 60

            # for weird pattern disruptions; 58 then 0 for example. 
            if values[0] - last_value < 0:
                for missing_value in range(last_value + 1, values[0]):
                    interpolated_row = [timestamp] + [int(missing_value), 0] + ["", ""]
                    csv_writer.writerow(interpolated_row)
                # Check if the fractional seconds have rolled over to 0
                if values[1] == 0 and last_value == 59:
                    values[0] = (values[0] + 1) % 60
            # Check for missing values and interpolate
            if values[0] - last_value > 0:
                for missing_value in range(last_value + 1, values[0]):
                    interpolated_row = [timestamp] + [int(missing_value), 0] + ["", ""]
                    csv_writer.writerow(interpolated_row)

            # Check if the fractional seconds have rolled over to 0
            if values[1] == 0 and last_value == 59:
                values[0] = (values[0] + 1) % 60

            # Write the original row with commas between values
            csv_writer.writerow([timestamp] + [int(value) if value != '' else '' for value in values])
            last_time, last_value = timestamp, int(values[0])
            
            # Check if the fractional seconds have rolled over to 0
            if values[1] == 0 and last_value == 59:
                values[0] = (values[0] + 1) % 60
    
# Call the function to interpolate the data
interpolate_data(csv_filename, output_csv_filename)

# Make a copy of the hz interpolated file
# Make the folder
interpolated_folder = "hz_interpolated_data"
os.makedirs(interpolated_folder, exist_ok=True)

# Make a copy of the hz interpolated file
duplicated_csv_filename2 = f"hz_interpolated_eye_status_log_duplicate_{timestamp}.csv"
shutil.copy(output_csv_filename, os.path.join(interpolated_folder, duplicated_csv_filename2))
print(f"Interpolated CSV file duplicated with timestamp: {duplicated_csv_filename2}")