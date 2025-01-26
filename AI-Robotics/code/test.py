from imports import *

# model = YOLO('./yolov8n.pt')
# results = model.predict(source='./potato/potato.jpg', conf=0.25)

# # Bounding box coordinates
# print(results[0].boxes.xyxy)

# # Detection confidence score
# print(results[0].boxes.conf)

# # 
# print(results[0].boxes.cls)

# roboflow.login()

# api_key = "vZJtxjMp8um8HhePA855"  # Replace MY-KEY with your actual Roboflow API key
# rf = roboflow.Roboflow(api_key=api_key)

# # Ensure the model_id matches exactly what's specified in your Roboflow account
# model_id = "potato-detection-3et6q/11"
# model = get_model(model_id)  # This function might vary based on the actual library methods available


# results = model.infer("./potato/potato.jpg")
# print(results)



input_data = tf.ones(shape=(8, 224, 224, 3))

# Pretrained backbone
model = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xl_backbone_coco"
)
output = model(input_data)

# Randomly initialized backbone with a custom config
model = keras_cv.models.YOLOV8Backbone(
    stackwise_channels=[128, 256, 512, 1024],
    stackwise_depth=[3, 9, 9, 3],
    include_rescaling=False,
)
output = model(input_data)