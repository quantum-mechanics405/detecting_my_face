import os
import cv2
import numpy as np

# Input folder containing faces
input_folder = 'in_image'  # Folder with your cropped face images
augmented_folder = 'out_aug_image'  # Output folder for augmented images

# Create the output folder if it doesn't exist
if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)

def augment_image(image):
    augmented_images = []

    # Rotation
    for angle in [0, 20, 40]:  # You can modify the angles
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)

    # Width and height shifts
    for dx, dy in [(int(w * 0.15), 0), (-int(w * 0.15), 0), (0, int(h * 0.15)), (0, -int(h * 0.15))]:
        # Translate image
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(shifted)

    # Shear
    for shear in [0.2, -0.2]:  # Shear factors
        M = np.array([[1, shear, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(sheared)

    # Zoom
    for scale in [1.2, 0.8]:  # Zoom in and out
        zoomed = cv2.resize(image, None, fx=scale, fy=scale)
        augmented_images.append(zoomed)

    # Horizontal Flip
    flipped = cv2.flip(image, 1)  # 1 means horizontal flip
    augmented_images.append(flipped)

    return augmented_images

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        
        # Load the image
        img = cv2.imread(image_path)
        
        # Augment the image
        augmented_images = augment_image(img)

        # Save the augmented images
        for i, aug_img in enumerate(augmented_images):
            aug_filename = f'aug_{i}_{filename}'  # Naming the augmented images
            aug_image_path = os.path.join(augmented_folder, aug_filename)
            cv2.imwrite(aug_image_path, aug_img)

print('Data augmentation complete! Augmented images saved in', augmented_folder)
