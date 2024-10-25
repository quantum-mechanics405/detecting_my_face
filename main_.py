import cv2
from mtcnn import MTCNN
import os

# Input folder containing multiple images
input_folder = 'imagess'
output_folder = 'output_faces'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the MTCNN face detector
detector = MTCNN()

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter for image files
        image_path = os.path.join(input_folder, filename)
        
        # Read the image
        img = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = detector.detect_faces(img_rgb)

        # Loop through all detected faces in the image
        for i, face in enumerate(faces):
            # Get the bounding box for each face
            x, y, width, height = face['box']
            
            # Crop the face from the image
            face_img = img_rgb[y:y+height, x:x+width]
            
            # Save the cropped face image
            face_path = os.path.join(output_folder, f'{filename}_face_{i+1}.jpg')
            cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            print(f'Saved face {i+1} from {filename} at {face_path}')
        
        # Clear the image variable to free up memory
        del img, img_rgb, faces

print('All faces have been saved from all images!')
