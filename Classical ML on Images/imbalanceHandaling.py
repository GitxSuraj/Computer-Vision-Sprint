# Precision & Recall
"""
-> True Positive(TP) : Model correctly predicts defect
-> True Negatives (TN) : Model correctly predicts the normal part.
-> False Postives (FP) : Model predicted a defect, but it was actually normal - False Alarm!
-> False Negatives (FN) : Model predicted normal, but it was actually defect - Missed Target!
"""
# From these - Two Crucial industry metrics:
# Precision 
"""
                TP
precision = ----------      -> Out of all model claimed defects - how many were actually defects?
             TP + FP

             
*High precision means low false alarm*
"""
# Recall - Sensitivity
"""
              TP
Recall = -------------       -> Out of all actual defects - how many model manage to catch?
            TP + FN


*High recall - missed very few targets*        
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# genrate highly imbalanced dataset - 1000 image features - 950 Normal - only 50 are defects
X_normal = np.random.normal(loc=0.0, scale=1.0, size=(950,64))
X_defect = np.random.normal(loc=0.5, scale=1.0, size=(50,64)) # Slightly diffrent distribution

X = np.vstack((X_normal, X_defect))
y= np.array([0]*950 + [1]*50)

#Split into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Note: stratify=y ensures the 95:5 split is preserved in both train and test sets!

# Lazy Native Model - No balancing
naiveModel = RandomForestClassifier(random_state=42)
naiveModel.fit(X_train, y_train)
naivePreds = naiveModel.predict(X_test)
print(f"Niave model Report\n{classification_report(y_test, naivePreds, zero_division=0)}")

# An Engineered Model - With class balancing
# class_weight = balanced automattical penalizes the model heavily
balancedModel = RandomForestClassifier(class_weight='balanced', random_state=42)
balancedModel.fit(X_train, y_train)
balancedPreds = balancedModel.predict(X_test)
print(f"Balanced model Report\n{classification_report(y_test, balancedPreds)}")
