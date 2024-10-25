import cv2
from mtcnn import MTCNN
from PIL import Image, ExifTags
import os
import gc
import numpy as np

# Input folder containing multiple images
input_folder = 'all_picturs'
output_folder = 'output_faces'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the MTCNN face detector
detector = MTCNN()

def correct_image_orientation(image):
    try:
        # Get the image's EXIF data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If there's no EXIF data or no orientation tag, do nothing
        pass

    return image

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter for image files
        image_path = os.path.join(input_folder, filename)
        
        # Read the image
        img = Image.open(image_path)
        # Correct the image orientation if necessary
        img = correct_image_orientation(img)

        # Resize the image while maintaining aspect ratio
        img.thumbnail((640, 480))  # This now works because img is a PIL Image

        # Convert the image to RGB format for MTCNN
        img_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Detect faces in the image
        faces = detector.detect_faces(img_rgb)

        # Check if any faces were detected
        if not faces:
            print(f'No faces detected in {filename}. Skipping to next image.')
            continue  # Skip to the next image if no faces are detected

        # Loop through all detected faces in the image
        for i, face in enumerate(faces):
            # Get the bounding box for each face
            x, y, width, height = face['box']
            
            # Crop the face from the image
            face_img = img_rgb[y:y + height, x:x + width]
            
            # Save the cropped face image
            face_path = os.path.join(output_folder, f'{filename}_face_{i + 1}.jpg')
            cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            print(f'Saved face {i + 1} from {filename} at {face_path}')
        
        # Clear variables and collect garbage to free memory
        del img, img_rgb, faces  # No need to delete face_img as it's defined inside the loop
        gc.collect()

print('All faces have been saved from all images!')
