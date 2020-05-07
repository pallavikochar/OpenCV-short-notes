# OpenCV-short-notes
## 10. cv.split, cv.merge, cv.resize, cv.add, cv.addWeighted, ROI
```
print(img.shape) #returns a tuple of number of rows, coloumns, and channels

print(img.size) #returns total number of pixels is accessed

print(img.dtype) #returns image datatype is obtained

cv2.split(img) #to split image in three channels i.e,b,g,r

cv2.merge((b,g,r)) #to merge the three channels into an image
ROI- region of interest
copy a region and paste at another place

cv2.add(src1,src2,dst,mask,dtype= -1) 
#calculates the per-element sum of two arrays or an array and a scalar. 
#src-source, dst-destination 

cv2.resize(img, (512,512)) 
#to make b0th the images of same size.(512,512) are no. of rows and coloumns

cv2.addWeighted(src1,alpha,src2,beta,gamma,dst,dtype) 
#calculates weighted sum of two arrays. 
#alpha beta are weights and gamma is a scalar
```



## 11. Bitwise Operations (bitwise AND, OR, NOT and XOR)
bitwise operatioons can be vety useful when working with masks

masks are binary images that indicates the pixel in which an operation is to be performed
```
cv2.bitwise_and(src1,src2,dst,mask) 
#have a look at truth table of the logic AND. 
#00-0,01-0,10-0,11-1. black as zeros and white as one

cv2.bitwise_or(src1,src2,dst,mask) 
#00-0,01-1,10-1,11-1.

cv2.bitwise_xor(src1,src2,dst,mask) 
#00-0,01-1,10-1,11-0.

cv2.bitwise_not(src,dst,mask) 
#0-1,1-0.
```



## 12. How to Bind Trackbar To OpenCV Windows
trackbars are useful whenever you want to change some value in your image dynamically at run time 
```
cv2.namedWindow('image') #to create a window with a name
cv2.createTrackbar(trackbarName, windowName, value, count, onChange) 
cv2.getTrackbarPos(trackbarName, windowName) #get the position of trackbars
```
How to add switch(on/off or switch from colored img to gray)  to trackbars




## 13.  Object Detection and Object Tracking Using HSV Color Space

HSV(Hue, Saturation, Value)-to separate image luminance from color information.
Hue corresponds to the color components (base pigment), hence just by selecting a range of Hue you can select any color (0-360)
Saturation is the amount of color (depth of the pigment)(dominance of Hue)(0-100%)
Value is basically the brightness of the color (0-100%)  
[look at this- C:\Users\Pallavi\Desktop\HSV color pallette.png]
```
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
#to change the color format from bgr to hsv 

lowerRangeBlue = np.array([110, 50, 50]) 
upperRangeBlue = np.array([130, 255, 255]) 
mask = cv2.inRange(hsv, lowerRangeBlue, upperRangeBlue) 
result = cv2.bitwise_and(frame-src1, frame-src2, mask = mask) 
```
Use trackbars for lower and upper bounds of hsv
you can also track an object while capturing a video.




## 14. Simple Image Thresholding
Thresholding a very popular segmentation technique used for separating an object from its background. Process of threhsholding involves comparing each pixel of an image with a predefined threshold value and this type of comparison of each pixel of an image to a threshold value divides all the pixels of the input image into two groups. First group involves the pixels having intensity value lower than threshold value and second group involves the pixels having intensity value greater than threshold value. Using different thresholding technique available in opencv we can give different value to these pixels which have higher and lower value than the threshold value.
```
cv2.threshold(src,thresh,maxval,type,dst) 
type- 1] cv2.THRESH_BINARY
2] cv2.THRESH_BINARY_INV
3] cv2.THRESH_TRUNC
4] cv2.THRESH_TOZERO #if pixel value is less than threshold, value assigned 
to pixel is 0. Else pixel value remain the same.
5] cv2.THRESH_TOZERO_INV
```



## 15. Adaptive Thresholding
Adaptive thresholding is a method where the threshold value is calculated for smaller region. 
```
cv2.adaptiveThreshold(src, maxval, adaptiveMethod, thresholdType,blockSize) 
#maxval is non-zero value assigned to pixels for which condition is satisfied.

adaptiveMethod- 1] cv2.ADAPTIVE_THRESH_MEAN_C
2] cv2.ADAPTIVE_THRESH_GAUSSIAN_C
```



## 16. matplotlib with OpenCV [some installation things]
matplotlib is a plotting library for python which gives you wide variety of plotting methods. visit matplotlib.org for more information
```
plt.imshow(src) 
plt.show() 
```
matplotlib reads image in rbg format and opencv reads image in bgr format. So, we need to convert bgr into rgb to see same image.
```
plt.xticks([]), plt.yticks([]) 

titles=['original image', 'binary', 'binaryinv', 'trunc', 'tozero', 'tozeroinv']
images=[img,th1,th2,th3,th4,th5,th6]
plt.subplot(noOfRows,noOfColumns, index of the image), plt.imshow(image[i], 'gray') 
plt.title(titles[i]  ) 
```
We can include multiple images into one window.




## 17. Morphological Transformations
Morphological transformations are some simple operations based on the image shape. They are normally performed on binary images. Two things required during these transformations are the original image and structuring element or a kernel which decides the nature of operation. Kernel tells you how to change the value of any given pixel by combining it with different amounts of the neighbouring pixels.
```
kernal = np.ones((2,2), np.uint8) #square of 2*2.bigger the size better the 
dilation but there's a problem-white area increases.
dilation = cv2.dilate(mask, kernal, numberOfIterations)
erosion = cv2.erode(mask,kernal, NumberOfIterations) 

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernal) 
#erosion followed by dilation.

closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernal) 
#dilation followed by erosion

gradient = cv2.morphologyEx(mask,cv2.MORPH_GRADIENT, kernal) 
#difference between dilation of erosion of image

tophat = cv2.morphologyEx(mask,cv2.MORPH_TOPHAT,kernal) 
#difference between image and opening of image
```



## 18. Smoothing Images | Blurring Images OpenCV
Most commonly used smoothing opoeration is to remove noise in the images. We can use diverse linear filters for smooothing. They are easy to achieve and are relatively fast. Various filters available in opencv are homogeneous,gaussian, Median, Bilateral filter
Homogenous filter is the most simple filter, each output pixel is the mean of its kernel neighbors.
In image processing, a kernel, convolution matrix, or mask is a small matrix. It is used for blurring, sharpening, embossing, edge detection, and more.
K = 1/K2*[matrix of ones-size-K[height]*K[width]] # K2 = K[width]K[height]

eg. for 5*5 kernel, 
```
kernel = np.ones((5, 5), np.float32)/25 
dst = cv2.filter2D(img, -1, kernel) 
```
As in one-dimensional signals, images also can be filtered with various low-pass filters(LPF), high-pass filters(HPF) etc.
LPF helps in removing noises, blurring the images.
HPF filters helps in finding edges in the images. 
```
blur = cv2.blur(img, (5, 5)) 
```
Gaussian filter is nothing but using different-weight-kernel, in both x and y direction. S in the result, pixels located in the middle of the kernel have higher weight and pixels located at the side have lower weight.

eg. for 5*5 kernel, kernel
```
gblur = cv2.GaussianBlur(img, (5, 5), 0)  
```
Median filter is something that replace each pixel's value with the median of its neighboring pixels. This method is great when dealing with 'salt and pepper noise'.
``` 
median = cv2.medianBlur(img, 5) #kernel size must be odd here

bilateralfilter = cv2.bilateralFilter(img, 9, 75, 75) 
#borders are preserved i.e, are kept sharp while rest image is blurred.
```



## 19. Image Gradients and Edge Detection
An image gradient is a directional change in the intensity or color in an image.
Image gradient methods- laplacian derivatives,
```
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=1) 
lap = np.unit8(np.absolute(lap)) 
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0) 
#change in direction in the intensity is in X direction-vertical

sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1) 
#change in direction in the intensity is in Y direction- horizontal

sobelX = np.unit8(np.absolute(sobelX)) 
sobelY = np.unit8(np.absolute(sobelY)) 

sobelCombined = cv2.bitwise_or(sobelX, sobelY) 
```



## 20. Canny Edge Detection in OpenCV
Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. It was developed by John F. Canny in 1986.

Canny edge detection algorithm is composed of 5 steps:
1. Noise reduction #apply gaussian filter to smooth
2. Gradient calculation #find intensity gradients of the image
3. Non-maximum suppression #to get rid of spurious response to edge detection
4. Double threshold #to determine the potential edges
5. Edge Tracking by Hysteresis #to finalize the detection of edges by suppressing all the other edges that are weak or not connected to strong edges
```
canny = cv2.Canny(img, threshold1, threshold2 )
```
use trackbars for better edge detection




## 21. Image Pyramids with Python and OpenCV 
Using image pyramids we just create the images of different resolutions and then we search for the object we want in all the images.
Pyramid , or pyramid representation, is a type of multi-scale signal representation in which a signal or an image is subject to repeated smoothing and subsampling.
Two types of image pyramids-
1] Gaussian pyramid #repeat filtering and subsampling of an image
Two funtioons available for gaussian pyramid which is called pyr down and pyr up
```
lr = cv2.pyrDown(img) #lower Resolution 
hr = cv2.pyrUp(img) #higher Resolution
layer = img.copy() #copies the image
gp = [layer] #gaussian pyramid
gp.append(layer) 
```
2] Laplacian pyramid #formed from gaussian pyramid. Theere is no exclusive function for creating laplacian pyramid.n  
A level of Laplacian Pyramid is formed by the difference between that level in Gaussian Pyramid and expanded version of its upper level in Gaussian Pyramid. 
```
lp = [layer] #laplacian pyramid
gaussianExtended = cv2.pyrUp(gp[i]) 
laplacian = cv2.subtract(gp[i-1],gaussianExtended) 
```



## 22. Image Blending using Pyramids in OpenCV
To know the shape of the image- img.shape
```
apple_orange = np.hstack((apple[:, :256], orange[:, 256:])) 
#but it is visible that one is apple and other is orange
```
1] Load the two images of apple and orange
2] Find the Gaussian Pyramids for apple and orange 
3] From Gaussian Pyramids, find their Laplacian Pyramids
4] Now join the left half of apple and righgt half of orange in each levels of Laplacian Pyramids
5] Finally from this joint image pyramids, reconstruct the original image.
```
apple_orange_pyramid = []
n =0
for  apple_lap, orange_lap in zip(lp_apple, lp_orange):
	n = n+1
	cols, rows, ch = apple_lap.shape
	laplacian = np.hstack((apple_lap[:, 0:int(cols/2)], orange_lap[:, int(cols/2):]))
	apple_orange_pyramid.append(laplacian)

#now reconstruct
apple_orange_reconstruct = apple_orange_pyramid[0]
for i in range(1, 6):
apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)  
apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)
```



## 23. Find and Draw Contours with OpenCV in Python
Contours can be explained as the curve joining all the continuous point along the boundary which are having the same color or intensity. Contour can be useful for shape analysis, object detection or object recognition.
```
contours, hierarchy = cv2.findContours(image, mode, method) 
#mode - cv2.RETR_TREE, method - cv2.CHAIN_APPROX_NONE
```
contours is a python list of all the contours in the iamge. Each individual contour is a Numpy array of (x, y) coordinates of boundary points of the object.  
 
```
cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
#contour index = -1 gives all contours
```





