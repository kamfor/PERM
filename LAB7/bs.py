#####################################################################
#Kamil Foryszewski, Jakub Wieczorek PERM L7
#####################################################################

# import freenect
import sys
import cv2
# import frame_convert2
import datetime
import numpy as np


def get_video():
    global camera
    # return frame_convert2.video_cv(freenect.sync_get_video()[0])
    retval, im = camera.read()
    return im


#####################################################################

def main():
    global camera
    camera = cv2.VideoCapture(0)


    # create background subtraction object

    mog = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=100, detectShadows=False);

    while (1):

        # if video file successfully open then read frame from video

        frame = get_video()

        # add current frame to background model and retrieve current foreground objects

        fgmask = mog.apply(frame);

        # threshold this and clean it up using dilation with a elliptical mask

        fgthres = cv2.threshold(fgmask.copy(), 0, 255, cv2.THRESH_BINARY)[1];

        #mask for circle detection

        mask = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (120, 120)), iterations=1); 

        # image for contour detecting

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # usrednianie gauss
        img_out = cv2.GaussianBlur(gray,(5,5),0)

        gray = img_out.astype("float"); 

        # krawedzie x
        kernel = np.array((
            [1, 0,-1],
            [2, 0, -2],
            [1, 0, -1]), dtype="int")
        img_out_x = cv2.filter2D(gray, -1, kernel)
        
        # krawedzie y
        kernel = np.array((
            [1, 2,1],
            [0, 0, 0],
            [-1, -2, -1]), dtype="int")
        img_out_y = cv2.filter2D(gray, -1, kernel)
        
        img_out = np.sqrt(np.power(img_out_x,2)+np.power(img_out_y,2))
        
        img_out = img_out/np.max(img_out)
        
        ret,img_out= cv2.threshold(img_out,0.23,1.0,cv2.THRESH_BINARY)
        
        img_8 = (img_out*255).astype('uint8')

        # apply move mask to contour image

        res = cv2.bitwise_and(img_8,img_8,mask = mask)

        # find circles

        circles = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT, 1, 20, param1=60, param2=40, minRadius=30, maxRadius=400)

        if circles != None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 5);


        cv2.imshow('detected circles', frame);
        cv2.imshow('movement mask', mask);
        cv2.imshow('detected contours', res)

        key = cv2.waitKey(30) & 0xFF; 

        if (key == ord('x')):
            keep_processing = False;

        if (key == ord('b')):
            mog = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=100, detectShadows=True);

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
