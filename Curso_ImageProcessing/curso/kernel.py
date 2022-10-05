import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

img = cv.imread("044R_3.png.png")
plt.imshow(img)
plt.show()


grey = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
plt.imshow(grey,cmap="gray")
plt.show()

kernel = np.ones((5,5),np.uint8)
# Blurring and erasing little details
grey = cv.GaussianBlur(grey,(9,9),0)
grey = cv.morphologyEx(grey, cv.MORPH_OPEN, kernel)
grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)
plt.imshow(grey,cmap="gray")
plt.show()


#Thresholding to highlight the more dark areas
grey = cv.threshold(grey,40,255,cv.THRESH_BINARY_INV)[1]
grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)
plt.imshow(grey,cmap="gray")
plt.show()

canny = cv.Canny(grey,100,200)
plt.imshow(canny,cmap="gray")
plt.show()

circles = cv.HoughCircles(grey,
                          cv.HOUGH_GRADIENT,
                          dp=1.1,
                          minDist=300,
                          param1=200,
                          param2=40,
                          minRadius=50,
                          maxRadius=400)
print(circles)

# Changing the dtype  to int
circles = np.uint16(np.around(circles))
cimg = img.copy()
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 10)

plt.imshow(cimg)