import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


# Load TFLite models

plant_interpreter = tflite.Interpreter(model_path="plant_model.tflite")
aphid_interpreter = tflite.Interpreter(model_path="aphid_model.tflite")
plant_interpreter.allocate_tensors()
aphid_interpreter.allocate_tensors()

# Class labels
plant_classes = ["Healthy", "Black Rot", "Light Blight", "Esca"]
aphid_classes = ["No Aphid", "Myzus persicae", "Aphis gossypii",
                 "Macrosiphum euphorbiae", "Aphis spiraecola"]


# Prediction function

def predict_leaf_from_image(img, plant_size=(224, 224), aphid_size=(224, 224)):
    """
    Predict plant disease and aphid species from a PIL Image.
    Aphid detection runs only if plant is Diseased.
    """
    # Prepare plant model input
    plant_img = img.resize(plant_size)
    plant_array = np.expand_dims(np.array(plant_img)/255.0, axis=0).astype(np.float32)

    input_details = plant_interpreter.get_input_details()
    output_details = plant_interpreter.get_output_details()
    plant_interpreter.set_tensor(input_details[0]['index'], plant_array)
    plant_interpreter.invoke()
    plant_probs = plant_interpreter.get_tensor(output_details[0]['index'])[0]

    healthy_prob = plant_probs[0]
    diseased_prob = sum(plant_probs[1:])
    most_likely_disease_idx = np.argmax(plant_probs[1:]) + 1
    most_likely_disease_label = plant_classes[most_likely_disease_idx]

    if diseased_prob > healthy_prob:
        plant_overall_label = "Diseased"

        # Prepare aphid model input
        aphid_img = img.resize(aphid_size)
        aphid_array = np.expand_dims(np.array(aphid_img)/255.0, axis=0).astype(np.float32)

        input_details = aphid_interpreter.get_input_details()
        output_details = aphid_interpreter.get_output_details()
        aphid_interpreter.set_tensor(input_details[0]['index'], aphid_array)
        aphid_interpreter.invoke()
        aphid_probs = aphid_interpreter.get_tensor(output_details[0]['index'])[0]
        aphid_pred_label = aphid_classes[np.argmax(aphid_probs)]
    else:
        plant_overall_label = "Healthy"
        aphid_probs = None
        aphid_pred_label = None

    return {
        "plant_overall": plant_overall_label,
        "healthy_prob": healthy_prob,
        "diseased_prob": diseased_prob,
        "most_likely_disease": most_likely_disease_label,
        "plant_class_probs": dict(zip(plant_classes, plant_probs)),
        "aphid_class": aphid_pred_label,
        "aphid_class_probs": dict(zip(aphid_classes, aphid_probs)) if aphid_probs is not None else None
    }


# Run predictions from folder

image_folder = "/home/pi/test_images/"  

for file in os.listdir(image_folder):
    if file.endswith((".jpg", ".png")):
        img_path = os.path.join(image_folder, file)
        img = Image.open(img_path).convert('RGB')
        result = predict_leaf_from_image(img)
        
        print(f"\nPrediction for {file}:")
        print(f"  Plant Status: {result['plant_overall']}")
        print(f"    Healthy prob: {result['healthy_prob']*100:.2f}%")
        print(f"    Diseased prob: {result['diseased_prob']*100:.2f}%")
        if result['plant_overall'] == "Diseased":
            print(f"    Most likely disease: {result['most_likely_disease']}")
            print(f"    Aphid Prediction: {result['aphid_class']}")


# Camera-ready code (future)

# import cv2
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     result = predict_leaf_from_image(img)
#     print(result)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
