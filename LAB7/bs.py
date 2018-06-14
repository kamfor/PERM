#####################################################################

# Example : perform GMM based foreground/background subtraction from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import sys
import freenect
import cv2
import frame_convert2
import datetime
import numpy as np

def get_video():
	return frame_convert2.video_cv(freenect.sync_get_video()[0])

#####################################################################

keep_processing = True;
camera_to_use = 0;

#####################################################################

# define video capture object

cap = freenect.sync_get_video()[0]

# check versions to work around this bug in OpenCV 3.1
# https://github.com/opencv/opencv/issues/6055

(major, minor, _) = cv2.__version__.split(".")
if ((major == '3') and (minor == '1')):
    cv2.ocl.setUseOpenCL(False);

# define display window name

windowName = "Live Camera Input"; # window name
windowNameBG = "Background Model"; # window name
windowNameFG = "Foreground Objects"; # window name
windowNameFGP = "Foreground Probabiity"; # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if 1:

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameBG, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameFG, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL);

    # create GMM background subtraction object (using default parameters - see manual)

    mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True);

    while (keep_processing):

        # if video file successfully open then read frame from video

        frame = get_video()

        # add current frame to background model and retrieve current foreground objects

        fgmask = mog.apply(frame);

        # threshold this and clean it up using dilation with a elliptical mask

        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1];
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3);

        # get current background image (representative of current GMM model)

        bgmodel = mog.getBackgroundImage();

        # display images - input, background and original



        img = fgdilated.copy()
        gray = img.astype("float"); 

        # usrednianie oknem 5x5
        kernel = np.ones((5,5), dtype="float") / 25.0
        img_out = cv2.filter2D(gray, -1, kernel)

        # krawedzie x
        kernel = np.array((
            [1, 0,-1],
            [2, 0, -2],
            [1, 0, -1]), dtype="int")
        img_out_x = cv2.filter2D(img_out, -1, kernel)
        
        # krawedzie y
        kernel = np.array((
            [1, 2,1],
            [0, 0, 0],
            [-1, -2, -1]), dtype="int")
        img_out_y = cv2.filter2D(img_out, -1, kernel)
        
        img_out = np.sqrt(np.power(img_out_x,2)+np.power(img_out_y,2))
        
        img_out = img_out/np.max(img_out)
        
        ret,img_out= cv2.threshold(img_out,0.3,1.0,cv2.THRESH_BINARY)
        
        img_8 = (img_out*255).astype('uint8')
        
        circles = cv2.HoughCircles(img_8,cv2.HOUGH_GRADIENT,1,40,param1=50,param2=35,minRadius=50,maxRadius=500)
       
        if circles != None:
            circles = np.uint16(np.around(circles))
        
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(fgdilated,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(fgdilated,(i[0],i[1]),2,(0,0,255),3)

        
        
        cv2.imshow(windowName,frame);
        cv2.imshow(windowNameFG,fgdilated);
        cv2.imshow(windowNameFGP,fgmask);
        cv2.imshow(windowNameBG, bgmodel);
        cv2.imshow('detected circles',frame)
        # start the event loop - essential


        key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False;
        
        if (key == ord('b')):
            mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True);

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.");
