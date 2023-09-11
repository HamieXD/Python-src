import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import os
import pygame
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

root = tk.Tk()

root.config(bg='light blue')
root.title('Hamiekung App')

main_frame = Frame(root)
main_frame.grid()
main_frame.config(bg='light blue')
main_frame.place(in_=root, anchor="c", relx=.5, rely=.5)

title_label = Label(main_frame,bg="light blue", font="Times 36", text="Drowsiness Detection")
title_label.grid(row=0, column=0, columnspan=2)
description_label = Label(main_frame,bg="light blue", font="Times 18", text="By SMET Surawittayakarn School")
description_label.grid(row=1, column=0, columnspan=2, pady=10)

middle_frame = Frame(main_frame,bg="light blue")
middle_frame.grid(row=2, column=0)

camera_label = Label(middle_frame)
camera_label.grid(row=0, column=0)

notify_label = Label(middle_frame,bg="white", font='Times 24')
notify_label.grid(row=1, column=0)

exit_button = Button(middle_frame, text = "Exit",width=15, height=3, command = root.destroy)
exit_button.grid(row=2, column=0, columnspan=2, pady=30, sticky=S)

# Constants
EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold for detecting drowsiness
EAR_CONSEC_FRAMES = 16  # Number of consecutive frames the EAR must be below the threshold to consider drowsiness
ALARM_SOUND_PATH = "alarm_sound.wav"  # Relative path to the alarm sound file

# Get the directory path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the shape predictor file
shape_predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Start video capture
video_capture = cv2.VideoCapture(1)

# Check if video capture is successfully opened
if not video_capture.isOpened():
    print("Failed to open the video capture.")
    exit()

# Initialize frame counters and drowsiness flag
frame_counter = 0
drowsy = False
alarm_playing = False

# Get the absolute path to the alarm sound file
alarm_sound_path = os.path.abspath(os.path.join(script_dir, ALARM_SOUND_PATH))

# Initialize pygame mixer
pygame.mixer.init()

# Load the alarm sound
pygame.mixer.music.load(alarm_sound_path)

def main():
    global drowsy
    global left_eye
    global right_eye
    global alarm_playing
    global frame_counter
    # Read frame from video capture
    ret, frame = video_capture.read()

    # Check if frame is successfully read
    if not ret:
        print("Failed to read the frame.")
        root.destroy()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        # Detect faces in the grayscale frame
        faces = detector(gray)

        # Iterate over detected faces
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Extract left and right eye landmarks
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], np.int32)
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], np.int32)

            # Calculate eye aspect ratios (EAR)
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below the threshold
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSEC_FRAMES:
                    if not alarm_playing:
                        drowsy = True
                        alarm_playing = True

                        # Play the alarm sound
                        pygame.mixer.music.play()

                    notify_label.configure(text=" - Drowsy - ", fg="red")
                    #cv2.putText(frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_counter = 0
                drowsy = False

        # If drowsiness is no longer detected, stop the alarm sound
        if not drowsy and alarm_playing:
            notify_label.configure(text=" - Not Drowsy - ", fg="red")
            pygame.mixer.music.stop()
            alarm_playing = False

    except Exception as e:
        print("Error:", str(e))

    # Display the resulting frame
    cv2image= cv2.cvtColor(video_capture.read()[1],cv2.COLOR_BGR2RGB)
    imgcv = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = imgcv)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
    camera_label.after(20, main)
    cv2.waitKey(1)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.destroy()

main()
root.attributes('-fullscreen', True)
notify_label.configure(text=" - Not Drowsy - ", fg="red")
root.mainloop()
# Release the video capture
video_capture.release()

# Stop pygame mixer
pygame.mixer.quit()

#By Thanapat and Techin (4,8)
