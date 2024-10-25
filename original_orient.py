from PIL import Image, ExifTags
import matplotlib.pyplot as plt

# Function to correct image orientation based on EXIF data
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

# Open an image file
image_path = 'pb1\IMG_20220520_174009.jpg'  # Replace with your image path
img = Image.open(image_path)

# Correct the image orientation if necessary
img = correct_image_orientation(img)

# Rotate the image 90 degrees clockwise
# rotated_img = img.rotate(-90, expand=True)

# Make a copy of the rotated image for resizing
resized_img = img.copy()

# Resize the image while maintaining aspect ratio
resized_img.thumbnail((640, 480))

# Display the original and resized images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display original image with corrected orientation
ax[0].imshow(img)
ax[0].set_title("Original Image (Corrected Orientation)")
ax[0].axis('off')

# Display resized image with maintained aspect ratio
ax[1].imshow(resized_img)
ax[1].set_title("Rotated and Resized Image (Aspect Ratio Kept)")
ax[1].axis('off')

plt.show()
