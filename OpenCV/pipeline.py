# will be wrapping up week 1
# End-to-End Production Preprocessing Pipeline
# Data Augmentation - Way of artificially expanding datset - before feeding it to neural network 
# How? - By creating variations of our existing images - like - flipping, rotating, altering brightness
# Why? - To Avoid Overfitting - feeding it only 1000 images - it will quicly memorize & Fail in real world
import numpy as np
import cv2

def preprocessImg(imgPath, targetSize = (244,244), augment=False):
    """
    industry standard image pre-processing pipeline for DL
    Takes - File path -> Processes the Image -> returns - PyTorch-ready array.
    """
    # Load the image using OpenCV
    img = cv2.imread(imgPath)
    if img is None:
        raise FileNotFoundError("Could not load image at", imgPath)
    
    # Fix BGR to RGB color spcae swap
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optional Data Augmentation - Only used during training - not evaluation
    if augment:
        # Randomly decide to horizontally flip the image (50% chance)
        if np.random.rand()>0.5:
            img = cv2.flip(img, 1)

        brightnessFactor = np.random.uniform(0.8,1.2) # - randomly adjust brightness by a small factor
        # Using np.clip ensures pixel values stay strictly between 0 & 255 - np.clip(Array, min, max) - if lower than min- makes it min, if higher than max - makes it max
        img = np.clip(img * brightnessFactor, 0, 255).astype(np.uint8) #- if by any change any pixel gets multplied with brightnees factor and exceed 255 it prevents that by capping it to 255

    # Resize to target dimensions - OpenCV expects Width, Height
    img = cv2.resize(img, targetSize)

    # Normalize pixel values from [0,255] down to [0.0, 1.0]
    img = img / 255.0

    # Transpose Image for HWC to PyTorch's required CHW format
    imgReady = np.transpose(img, (2,0,1))

    return imgReady
#testing the pipeline
if __name__ == "__main__":
    testPath = "test.jpg"
    try:
        #Run pipeline with augementation turned on
        processedTensor = preprocessImg(testPath,targetSize=(244,244),augment=True)
        print(f"Pipeline Excetuion Succesfull!\nOutput Image Propertie:-\nShape:{processedTensor.shape}, Data Type:{processedTensor.dtype}, Value Range: {processedTensor.min()} to {processedTensor.max()}")
    except Exception as e:
        print("Pipeline Failed! Error:",e)