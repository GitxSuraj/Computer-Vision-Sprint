from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load REAL data (8x8 pixel images of digits 0 through 9)
digits = load_digits()
X_images = digits.images  # Shape: (1797, 8, 8)
y = digits.target         # Labels: 0, 1, 2... up to 9

# 2. Flatten the images so our classical model can read them
# (For tiny 8x8 images, we can flatten directly; no HOG needed yet!)
num_samples = len(X_images)
X_flat = X_images.reshape((num_samples, -1)) # Reshapes (1797, 8, 8) to (1797, 64)

print(f"Loaded {num_samples} real handwritten digits.")
print("Flat feature matrix shape:", X_flat.shape)

# 3. Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.3, random_state=42)

# 4. Train an SVM Classifier
classifier = SVC(kernel='rbf', C=1.0) # Using 'rbf' kernel for non-linear patterns
classifier.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = classifier.predict(X_test)
print("\n--- Real Model Evaluation ---")
print("Accuracy on Real Data:", accuracy_score(y_test, y_pred))