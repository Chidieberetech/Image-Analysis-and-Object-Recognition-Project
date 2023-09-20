
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image        ## if u can change the variables names would be better##
                        ## choose all the image we have ##
                        ## use for loop to load all the images##
image_path = r'C:\Users\LENOVO\Desktop\Weimar Digital Engineering\IAOR\project\Cracks\20230918_133059.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                ## try to make the background white and the crack white##
                ## use the code we used in assignment 1 and change the min, max values##
threshold_value = 70            # use different values of thresholding
_, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# Perform post-processing (optional)
kernel = np.ones((3, 3), np.uint8)
thresholded_image = cv2.erode(thresholded_image, kernel, iterations=1)
thresholded_image = cv2.dilate(thresholded_image, kernel, iterations=1)

# Visualize the original image and segmented result
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(122)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Crack Segmentation')

plt.show()


                        ### try to fix this. im not sure if what i have done is correct##
# Perform connected component analysis
num_labels, labeled_image = cv2.connectedComponents(thresholded_image)

# Feature Engineering
features = []
labels = []

for label in range(1, num_labels):
    # Extract region properties
    region_mask = (labeled_image == label).astype(np.uint8)  # Create a binary mask for the current region
    area = cv2.countNonZero(region_mask)  # Count non-zero pixels as area

    # Calculate circularity
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0  # Use the contour's perimeter if available

    circularity = 0
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

    # You can add more feature calculations here

    # Classify based on rules (example rule: if circularity < 0.2, classify as crack)
    if circularity < 0.2:
        labels.append('crack')
    else:
        labels.append('no-crack')

# Visualize the connected components
plt.figure(figsize=(8, 6))
plt.imshow(labeled_image, cmap='jet')
plt.title(f'Connected Components: {num_labels - 1} Cracks Detected')  # Subtract 1 for the background
plt.colorbar()
plt.show()

# Print the labels for each region
for i, label in enumerate(labels):
    print(f'Region {i + 1}: {label}')
