import cv2
import numpy as np

def compute_crack_lengths(image_path, threshold_value=90, circularity_threshold=0.2):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform thresholding
    _, threshold_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the thresholded image to create a binary image
    binary_image = cv2.bitwise_not(threshold_image)

    # Perform connected component analysis
    num_labels, labeled_image = cv2.connectedComponents(binary_image)

    # Initialize a list to store crack lengths
    crack_lengths = []

    for label in range(1, num_labels):
        # Extract region properties
        region_mask = (labeled_image == label).astype(np.uint8)
        area = cv2.countNonZero(region_mask)

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0

        circularity = 0
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

        # Classify based on circularity
        if circularity < circularity_threshold:
            # Calculate the length of the crack region in pixels
            crack_lengths.append(perimeter)

    return crack_lengths

# Process and compute crack lengths for the test set
crack_lengths = []

for i in range(1, 28):
    filename = f'project_image_{i}.jpg'
    lengths = compute_crack_lengths(filename)
    crack_lengths.extend(lengths)

# Report the crack lengths
for i, length in enumerate(crack_lengths, start=1):
    print(f'Crack {i}: Length = {length:.2f} pixels')
