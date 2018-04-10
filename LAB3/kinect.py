#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import datetime
import numpy as np

cv2.namedWindow('Video')
print('Press ESC in window to stop')


def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

search_chessboard = False
do_hsv = False

while 1:
    img = get_video()
    img_out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #
    # Chessboard 
    #
    if search_chessboard:
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6,5), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
             corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
         
             # Draw and display the corners
             cv2.drawChessboardCorners(img_out, (6,5), corners2, ret)


    #
    # HSV threshold
    #
    if do_hsv:

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of color in HSV
        lower = np.array([0,0,70])
        upper = np.array([179,68,255])

        # Threshold the HSV image to get only selected colors
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

        cv2.imshow('Mask', mask)

        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        maxarea = -1
        maxcnt = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > maxarea and area > 10000:
                maxarea = area 
                maxcnt = cnt

        if maxarea > 0:
            cv2.drawContours(img_out, [maxcnt], 0, (0,255,0), 3)
            (x,y),radius = cv2.minEnclosingCircle(maxcnt)
            cv2.circle(img_out, (int(x), int(y)), int(radius) ,(0,0,255),2)
            #print(radius)
            Z = 521*(8/radius)
            
            X = (x-317)/521*Z
            Y = (y-258)/521*Z
            print("X: ", X, "Y: ", Y, "Z: ", Z)




    # display image
    cv2.imshow('Video', img_out)


    # wait for key press
    ch = cv2.waitKey(10)

    # nothing pressed
    if ch < 0:
        continue

    # ESC - end process
    if ch == 27:
        break
    # s - save picture
    if chr(ch) == 's':
        now = datetime.datetime.now()
        fname = now.strftime("%Y%m%d_%H%M%S%f.png");
        cv2.imwrite(fname, img)
        print("Saved: " + fname)

    # c - toggle search chessboard flag
    if chr(ch) == 'c':
        search_chessboard = not search_chessboard
        print("Chessboard searching: " + str(search_chessboard))

    # h - do hsv thresholding
    if chr(ch) == 'h':
        do_hsv = not do_hsv
        print("HSV thresholding: " + str(do_hsv))
