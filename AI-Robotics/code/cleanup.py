from imports import *

data = './withBackground/'

print(os.listdir(data))

# Removing incompatible data formats and corruted data
count = 0
image_extensions = ['.png', '.jpg']
img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]

for filepath in Path(data).glob('*'):

    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)

        if img_type is None:
            print(f"{filepath} is name an image")

        if img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by tensorflow")
            os.remove(filepath)
            count += 1

print(f"Removed {count} images")

print("\nLooking for incompatble images")

for filepath in Path(data).glob('*'):
    for image_name in os.listdir(filepath):
        # Construct the full path to the image file
        image_path = os.path.join(filepath, image_name)
        try:
            # Decode the image using TensorFlow
            decoded_image = tf.io.read_file(image_path)
            decoded_image = tf.image.decode_image(decoded_image)
        except tf.errors.InvalidArgumentError as e:
            # If decoding fails, print the error message
            print("Error decoding image:", e)
            print("Image path:", image_path)

print("All done :)")


