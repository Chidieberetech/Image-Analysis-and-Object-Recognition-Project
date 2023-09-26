import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# Create a function for crack segmentation
def segment_cracks(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    Pic_Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance the image using contrast stretching
    hist_min = np.min(Pic_Gray)
    hist_max = np.max(Pic_Gray)
    new_gray_value = ((Pic_Gray - hist_min) / (hist_max - hist_min)) * 255
    new_gray_value = np.uint8(new_gray_value)

    # Thresholding
    threshold_value = 90
    _, thresholded_image = cv2.threshold(Pic_Gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    thresholded_image = cv2.erode(thresholded_image, kernel, iterations=1)
    thresholded_image = cv2.dilate(thresholded_image, kernel, iterations=1)

    return thresholded_image  # Return the thresholded image

# Directory containing images

#change the path accordingly 
image_dir = r'C:\Users\mitre\Documents\Bauhaus University Weimar\IAOR 2023\IAOR 2023 Project\Task1–Data Engineering'

# List to store results
results = []

# Process each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        thresholded_image = segment_cracks(image_path)  # Get the thresholded image
        results.append((filename, thresholded_image))

# Create a PDF to save results
pdf_filename = 'crack_segmentation_results.pdf'

# Use PdfPages correctly
with PdfPages(pdf_filename) as pdf:
    # Visualize and save results
    for filename, thresholded_image in results:
        image = cv2.imread(os.path.join(image_dir, filename))
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(122)
        plt.imshow(thresholded_image, cmap='gray')  # Use thresholded_image
        plt.title('Crack Segmentation')

        pdf.savefig()
        plt.close()
