import numpy as np
#Dummy grayscale image matrix
matrix = np.array([
    [150,20,30],
    [40,200,10],
    [80,90,100]
], dtype=np.uint8)

# Have to isolate only bright Pixels (Value > 100)
# This creates the mask of True/False values

mask = matrix > 100
print("The Mask : \n", mask) # output as matrix of true false

#Use mask to modify only those specific pixels(True) 
matrix[mask]=255
print("\nModified Image\n", matrix)

# This is how green screen removel works
