# classical ml algo - like SVM or Random Forest -  usually fail if we throw raw pixel matrices dirctly intho them
# why? - a slight change in img shifts/changes the whole matrix - algo thinks it's a completely diffrent image
# fix - feature extraction - instead of raw colors - math structual patters 
# patterns? - Like Edges, gradients & Shapes -> feed this to ML models

# industry standard for classical vision features: HOG

# HOG - Histogram of Oriented Gradients
"""
What is HOG?
-Intution Over Formula
Matehamtically Extreacts Edges and contours
-> as we recongnize a borderd image by edges- it kind of does the same

* Gradient Calculation - looks at every pixel > check for drasrtic brightness changes > If dected it an EDGE
* Orientaton - calcultes which direction edge is pointing (vertical, horizonatal, diagonal)
* Histograms - divides img in small blocks - counts how many edges are pointing in which direction

-> Output - Simplified Vector of num - describes the shape of the object - ignoring lighting changes or bg colors
"""

# The Classical ML Vision Pipeline
"""
-> Scikit-Learn(sklearn) library
-> Simulate how industry pipeline take a image - Extracts HOG and trains SVM (Support Vector Machine)
"""

import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#100 images of shape(64,64,3)- Data set simulation - in production - will load actual image path via pandas 
numSamples = 100
mockImages = np.random.randint(0,256, size = (numSamples, 64,64,3), dtype=np.uint8)
#0 - Negative/BG, 1 = Target Image
mockLabels = np.random.choice([0,1], size=numSamples)
print(f"{numSamples} raw images loaded")

#Feature Extraction Loop
featureList = []
for img in mockImages:
    # mock image to grayscale for HOG extraction
    # RGBA/RGB -> Gray approximation using weights
    grayImg = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32) #Luminosity Formula
    ### The Code Breakdown
    """
    * `img[..., :3]`: The `...` is called an **Ellipsis**. It tells NumPy: *"Keep all dimensions before this."* For an image of shape `(64, 64, 3)`, it means keep the Height (64) and Width (64). The `:3` ensures we only pull the first three color channels (RGB), ignoring an alpha/transparency channel if it exists.
    * `np.dot(matrix, vector)`: This performs a **matrix dot product**. For every single pixel in your image, it takes the $[R, G, B]$ values, multiplies them by $[0.2989, 0.5870, 0.1140]$, and sums them up into a single grayscale value.
    * `.astype(np.float32)`: This converts the pixel values from integers into 32-bit floating-point numbers. The `skimage` HOG function requires floats to calculate smooth gradients.
    """
    #Extract HOG features
    #Orientation = 9 - Edge angles into 9 Buckets
    features = hog(grayImg,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    featureList.append(features)

X = np.array(featureList)
y = mockLabels
print("Extracted HOG feature vector shape", X.shape)
#Matrix in now flat and optimized for ML

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training the classifier - SVC
clf = SVC(kernel="linear", C=1.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred)*100,"%") # Accuracy (Approx 0.5)will be low coz it's random noise genrated, not real images
print("Classification Report:\n", classification_report(y_test,y_pred))
    