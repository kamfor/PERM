#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import datetime
import numpy as np

def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])

def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

def main():
    while 1:
        img = get_video().copy()
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray1.astype("float"); 

        # Convert BGR to HSV
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('detected circles',img)
        
			
        
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

if __name__=='__main__':
    main()


