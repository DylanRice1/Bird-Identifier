import os
import cv2
import numpy as np
import re
from imports import *  

# Load classifier model
classifier = load_model("./EfficientNetB0.keras")

# Class names list
class_names = ['Blackbird', 'Bluetit', 'Carrion_Crow', 'Chaffinch', 'Coal_tit', 'Dunnock', 'Feral Pigeon', 'Goldfinch', 'Great Tit', 'Greenfinch', 'House_Sparrow', 'Jackdaw', 'Long_Tailed_Tit', 'Magpie', 'Robin', 'Song_Thrush', 'Starling', 'Wood_Pigeon', 'Wren']

def run_detection():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(title='Select an Image', filetypes=[('Image Files', '*.jpg *.jpeg *.png')])
    if not file_path:
        print("No file selected.")
        return None

    model = YOLO('yolov8n.pt')  
    model.conf_thres = 0.25

    # Run detection on the specified image
    results = model.predict(source='./potato/Robin.jpg', save=True, save_txt=True, save_crop=True)
    return results

def get_latest_predict_directory(base_path='./runs/detect'):
    # Compile a regular expression to find directories that match the format
    pattern = re.compile(r'^predict(\d+)$')
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and pattern.match(d)]
    if directories:
        latest_directory = max(directories, key=lambda x: int(pattern.match(x).group(1)))
        latest_directory_path = os.path.join(base_path, latest_directory, 'crops/bird')
        return latest_directory_path
    else:
        return None

def load_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is not None:
        print("Image loaded successfully.")
    else:
        print("Failed to load image.")
    return image

def resize_and_normalize_image(image):
    # Resize image to the model input size
    resized_image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)
    # Normalize the image
    normalized_image = resized_image / 255.0
    return normalized_image

def prepare_image_for_model(image_path):
    image = load_image(image_path)
    if image is not None:
        image_ready = resize_and_normalize_image(image)
        # Add a batch dimension
        image_batch = np.expand_dims(image_ready, axis=0)
        return image, image_batch
    else:
        return None, None

def predict_and_annotate(image_path):
    image, image_batch = prepare_image_for_model(image_path)
    if image is not None:
        predictions = classifier.predict(image_batch)
        predicted_class_index = np.argmax(predictions)
        predicted_probability = np.max(predictions)
        predicted_class_name = class_names[predicted_class_index]

        # Annotate the image with the class name and probability
        label = f"{predicted_class_name}: {predicted_probability:.2f}"
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the annotated image
        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)  # Wait until a key is pressed to close the window
        cv2.destroyAllWindows()
    else:
        print("Image not loaded properly for prediction.")

def main():
    run_detection()  

    latest_predict_dir = get_latest_predict_directory()
    if latest_predict_dir:
        files = os.listdir(latest_predict_dir)
        if files:
            first_image_file = files[0]
            first_image_path = os.path.join(latest_predict_dir, first_image_file)
            predict_and_annotate(first_image_path)
        else:
            print("No images found in the directory.")
    else:
        print("No prediction directories found.")

if __name__ == '__main__':
    main()
