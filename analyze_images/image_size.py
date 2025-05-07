import cv2

# Load the image
image_path = '/home/hpmcclea/SemiF-SyntheticPipeline/projects/second-try-bbox-blur-num-comp/mixed/results/images/1f329c50-b9ba-4188-8908-3b5770b017e9.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image. Check the path.")
else:
    height, width = image.shape[:2]
    print(f"Width: {width}px, Height: {height}px")
#
#Width: 1804px, Height: 1204px