from frames import dual_frame, approx, stack_blocks, init_blocks
from multiprocessing import Pool
import queue
import pyautogui
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import time
import numpy as np

# For webcam input:
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#IMG_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
#IMG_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
screen_width, screen_height = pyautogui.size()
   
pyautogui.PAUSE = 0
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    frame_idx = 0
    success, image = cap.read()
    # Vertical and horizontal indices
    L=10
    N=1
    #step=20
    step=L
    #s=0.1
    s=0.5
    block_idxs, vstack_idxs = init_blocks(image.shape, step, L)
    M, Mtilde, window = dual_frame(L=L,N=N,s=s)
    do_dual=False

    buffer_x = np.empty(3)
    buffer_y = np.empty(3)

    while cap.isOpened():
        old_image = image
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        #image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if do_dual:
            with Pool() as p:
                blocks = np.array( p.starmap( approx,
                    ([image[bx:bx+L, by:by+L,0], M, Mtilde, window] for bx, by in block_idxs)))
                stack_blocks(blocks, image[:,:,0], vstack_idxs, L)

                blocks = np.array( p.starmap( approx,
                    ([image[bx:bx+L, by:by+L,1], M, Mtilde, window] for bx, by in block_idxs)))
                stack_blocks(blocks, image[:,:,1], vstack_idxs, L)

                blocks = np.array( p.starmap( approx,
                    ([image[bx:bx+L, by:by+L,2], M, Mtilde, window] for bx, by in block_idxs)))
                stack_blocks(blocks, image[:,:,2], vstack_idxs, L)


        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            l = hand_landmarks.landmark[8]
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            #x = image.shape[1]*l.x
            #y = image.shape[0]*l.y
            #x = image.shape[1]*np.mean([l.x for l in hand_landmarks.landmark])
            #y = image.shape[0]*np.mean([l.y for l in hand_landmarks.landmark])
            buffer_x[frame_idx % buffer_x.shape[0]] = image.shape[1]*np.mean(
                    [l.x for l in hand_landmarks.landmark])
            buffer_y[frame_idx % buffer_x.shape[0]] = image.shape[0]*np.mean(
                    [l.y for l in hand_landmarks.landmark])
            x = np.mean(np.dot(buffer_x, [0.1, 0.2, 0.7]))
            y = np.mean(np.dot(buffer_y, [0.1, 0.2, 0.7]))
            image = cv2.circle(image, (int(x),int(y)),radius=4,thickness=8,color=(0,0,0))

            # Update cursor position every k frames.
            pyautogui.moveTo(
                    1.5*screen_width*(1-1/image.shape[1]*x),
                    1.3*screen_height/image.shape[0]*y,
                    duration = 0)
            frame_idx+=1

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        #cv2.imshow('mediapipe hands', cv2.flip((image+old_image)/2, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
cap.release()

