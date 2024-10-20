import numpy as np
import cv2
import time
import math
import sys
import argparse
from datetime import datetime
import uuid
import os
import shutil
from matplotlib import pyplot as plt
import glob
import requests

# Function to classify the similarity between two grayscale histograms
def classify_gray_hist(image1, image2, size=(256, 256)):
    # Resize the images to a fixed size
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    
    # Calculate the histogram for both images
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    
    # Plot the histograms for visual comparison (red and blue lines)
    plt.plot(range(256), hist1, 'r')
    plt.plot(range(256), hist2, 'b')
    plt.show()
    
    degree = 0
    # Calculate the degree of similarity between the histograms
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            # Calculate difference and normalize
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    # Normalize the degree of similarity
    degree = degree / len(hist1)
    return degree

# Function to calculate similarity between two grayscale images
def calculate(image1, image2):
    # Calculate the histogram for both images
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    
    degree = 0
    # Calculate the similarity between histograms
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            # Calculate difference and normalize
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    # Normalize the degree of similarity
    degree = degree / len(hist1)
    return degree

# Function to classify images based on average hash (aHash)
def classify_aHash(image1, image2):
    # Resize images and convert to grayscale
    image1 = cv2.resize(image1, (100, 100))
    image2 = cv2.resize(image2, (100, 100))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Generate hash for both grayscale images
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    
    # Calculate Hamming distance between the two hashes
    return Hamming_distance(hash1, hash2)

# Function to classify images based on perceptual hash (pHash)
def classify_pHash(image1, image2):
    # Resize images and convert to grayscale
    image1 = cv2.resize(image1, (100, 100))
    image2 = cv2.resize(image2, (100, 100))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Apply discrete cosine transform (DCT) to the grayscale images
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    
    # Extract the top-left 8x8 DCT region for comparison
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    
    # Generate hash for both DCT regions
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    
    # Calculate Hamming distance between the two hashes
    return Hamming_distance(hash1, hash2)

# Function to generate a binary hash from a grayscale image
def getHash(image):
    average = np.mean(image)  # Calculate average intensity of the image
    hash = []
    
    # Compare each pixel's intensity with the average
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > average:
                hash.append(1)
            else:
                hash.append(0)
    return hash

# Function to calculate Hamming distance between two hash values
def Hamming_distance(hash1, hash2):
    num = 0
    # Count the number of differing bits between the two hashes
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

# Function to calculate the variance of the Laplacian, used to measure sharpness (focus) of an image
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Function to stop process based on a condition
def stop(c):
    if c > 10:
        return False
    else:
        return True

# Start video capture from the camera
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 4096)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 2160)

n = 0
print('Turning on Camera...')
time.sleep(1)
print('Starting calibration process...\nPlease put files on the scanner then remove it')

# Remove previous files in the directory
filenames = glob.glob('/home/pi/Patient/' + r'/*')
for filename in filenames:
    os.remove(filename)

fm = []  # Array to store sharpness values for calibration
while True:
    n = n + 1
    ret, frame = cap.read()
    cv2.imwrite('/home/pi/Test/' + str(n) + '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY)])
    print('Calibrating')

    # Convert the captured image to grayscale and measure sharpness using variance of Laplacian
    imagePath = '/home/pi/Test/' + str(n) + '.jpg'
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm.append(variance_of_laplacian(gray))
    
    # Remove the image after processing
    os.remove('/home/pi/Test/' + str(n) + '.jpg')

    # Stop after taking 20 images for calibration
    if n == 20:
        break

time.sleep(2)
print('Calibration Completed\nNow please put files on the scanner')

# Sort and take the average sharpness of the top 10 calibrated images
fm.sort(reverse=True)
average = np.mean(fm[5:15])
print(str(average), str(fm))
time.sleep(5)

# Begin the scanning process
PicturePath_Mark = []
fm = []
n = 0
while True:
    n = n + 1
    ret, frame = cap.read()
    cv2.imwrite('/home/pi/Test/' + str(n) + '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('Working...')

    # Convert the captured image to grayscale and calculate sharpness
    imagePath = '/home/pi/Test/' + str(n) + '.jpg'
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm.append(variance_of_laplacian(gray))
    print('Solution:' + str(fm[-1]))

    # Discard blurry images
    if fm[-1] < average:
        print('This Photo is blurry. Ditching...')
        os.remove('/home/pi/Test/' + str(n) + '.jpg')
    else:
        # Save the non-blurry image
        Time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S.jpg')
        New_imagePath = '/home/pi/Patient/' + Time
        shutil.move(imagePath, New_imagePath)
        PicturePath_Mark.append(New_imagePath)

        # Compare the new image with the previous one
        if n > 1:
            image1 = PicturePath_Mark[-2]
            image2 = PicturePath_Mark[-1]
            img1 = cv2.imread(image1)
            img2 = cv2.imread(image2)
            num1 = classify_aHash(img1, img2)
            print('Difference:' + str(num1))
            
            # If the two images are too similar, discard the blurrier one
            if num1 < 500:
                if fm[-1] > fm[-2]:
                    os.remove(image1)
                    PicturePath_Mark.pop(-2)
                    fm.pop(-2)
                else:
                    os.remove(image2)
                    PicturePath_Mark.pop(-1)
                    fm.pop(-1)
            else:
                # If the images are different, upload to the server
                url = 'http://161.92.142.211:6543/imagecap/jiafeng'
                name = PicturePath_Mark[-2]
                fileinfo = {'realfile': open(name, 'rb')}
                data = {"realname": name}
                requests.post(url, files=fileinfo, data=data)
                print('good')
    time.sleep(0.5)
