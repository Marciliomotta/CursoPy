import numpy as np
import cv2 as cv
img = cv.imread('044R_3.png',0)
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg  ,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg  ,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()

circles = cv2.HoughCircles(sobel,
                          cv2.HOUGH_GRADIENT,
                          dp=1.1,
                          minDist=200,
                          param1=65,
                          param2=30,
                          minRadius=20,
                          maxRadius=300)