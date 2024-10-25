from PIL import Image
import matplotlib.pyplot as plt

# Open an image file
image_path = 'pb1\IMG_20220520_174619.jpg'  # Replace with your image path
img = Image.open(image_path)

# Rotate the image 90 degrees clockwise
rotated_img = img.rotate(-90, expand=True)

# Make a copy of the rotated image for resizing
resized_img = rotated_img.copy()

# Resize the image while maintaining aspect ratio
resized_img.thumbnail((640, 480))

# Display the original and resized images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display original image
ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[0].axis('off')

# Display resized image with maintained aspect ratio
ax[1].imshow(resized_img)
ax[1].set_title("Rotated and Resized Image (Aspect Ratio Kept)")
ax[1].axis('off')

plt.show()
