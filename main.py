import numpy as np 
import cv2 

# Read and preprocess the image
image = cv2.imread('digits.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Divide the image into 5000 segments of 20x20 pixels each
divisions = [np.hsplit(row, 100) for row in np.vsplit(gray_img, 50)]
NP_array = np.array(divisions)

# Prepare training and testing data
train_data = NP_array[:, :50].reshape(-1, 400).astype(np.float32)
test_data = NP_array[:, 50:].reshape(-1, 400).astype(np.float32)

# Create labels for the digits (0-9)
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = np.repeat(k, 250)[:, np.newaxis]

# Initialize and train kNN classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Predict using kNN classifier with k=3
ret, output, neighbours, distance = knn.findNearest(test_data, k=3)

# Calculate accuracy of the classifier
matched = output == test_labels
correct_OP = np.count_nonzero(matched)
accuracy = (correct_OP * 100.0) / output.size

# Display accuracy
print(f"Accuracy of kNN classifier: {accuracy:.2f}%")
