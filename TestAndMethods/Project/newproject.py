import numpy as np
import cv2
from matplotlib import pyplot as plt

# read input as grayscale
img = cv2.imread('028L_1.png')
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# convert image to floats and do dft saving as complex output
dft = cv2.dft(np.float32(cimg),flags = cv2.DFT_COMPLEX_OUTPUT)

# apply shift of origin from upper left corner to center of image
dft_shift = np.fft.fftshift(dft)

# extract magnitude and phase images
mag, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

# get spectrum for viewing only
spec = np.log(mag) / 30

# NEW CODE HERE: raise mag to some power near 1
# values larger than 1 increase contrast; values smaller than 1 decrease contrast
mag = cv2.pow(mag, 1.1)

# convert magnitude and phase into cartesian real and imaginary components
real, imag = cv2.polarToCart(mag, phase)

# combine cartesian components into one complex image
back = cv2.merge([real, imag])

# shift origin from center to upper left corner
back_ishift = np.fft.ifftshift(back)

# do idft saving as complex output
img_back = cv2.idft(back_ishift)

# combine complex components into original image again
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# re-normalize to 8-bits
min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
edges = cv2.Canny(img_back, 40,50)
kernel = np.ones((5,5),np.uint8)
sobel = cv2.Sobel(cimg, -1, 0, 1)

dilation = cv2.dilate(edges,kernel,iterations = 1)

circles = cv2.HoughCircles(sobel,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=40,minRadius=40,maxRadius=80)
circles = np.uint16(np.around(circles))

dimg = img.copy()
for i in circles[0,:]:
     #draw the outer circle
     cv2.circle(dimg,(i[0], i[1]), i[2], (0, 255, 0), 2)
     # draw the center of the circle
     cv2.circle(dimg,(i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles',dimg)

cv2.imshow("lena_grayscale_opencv.png", img)

fig, ax = plt.subplots(ncols=7,figsize=(20,10))
ax[0].imshow(img, cmap = 'gray')
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(cimg, cmap = 'gray')
ax[1].set_title('gray')
ax[1].axis('off')
ax[2].imshow(mag, cmap = 'gray')
ax[2].set_title('mag')
ax[2].axis('off')
ax[3].imshow(img_back, cmap = 'gray')
ax[3].set_title('img_back')
ax[3].axis('off')
ax[4].imshow(edges, cmap = 'gray')
ax[4].set_title('Canny')
ax[4].axis('off')
ax[5].imshow(sobel, cmap = 'gray')
ax[5].set_title('sobel')
ax[5].axis('off')
ax[6].imshow(dimg, cmap = 'gray')
ax[6].set_title('circle')
ax[6].axis('off')
plt.show()


# write result to disk
cv2.imwrite("lena_grayscale_opencv.png", img)
cv2.imwrite("../lena_grayscale_coefroot_opencv.png", img_back)

cv2.waitKey(0)
cv2.destroyAllWindows()