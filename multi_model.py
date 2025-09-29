import tensorflow as tf
import numpy as np
from PIL import Image

# Load models
plant_model = tf.keras.models.load_model("/Users/sharyudilipmagre/Internship Project/plant_effiecientnet.keras")
aphid_model = tf.keras.models.load_model("/Users/sharyudilipmagre/Internship Project/aphid_inception.keras")

# Class labels
plant_classes = ["Healthy", "Black Rot", "Light Blight", "Esca"]
aphid_classes = ["No Aphid", "Myzus persicae", "Aphis gossypii", 
                 "Macrosiphum euphorbiae", "Aphis spiraecola"]

def predict_leaf(image_path, plant_size=(224, 224), aphid_size=(224, 224)):
    """
    Predict plant disease and aphid species for a single image.
    Aphid detection runs only if plant is Diseased.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Plant model input
    plant_img = img.resize(plant_size)
    plant_array = np.array(plant_img) / 255.0
    plant_array = np.expand_dims(plant_array, axis=0)

    # Plant model prediction
    plant_probs = plant_model.predict(plant_array)[0]  # softmax probabilities
    healthy_prob = plant_probs[0]
    diseased_prob = sum(plant_probs[1:])
    most_likely_disease_idx = np.argmax(plant_probs[1:]) + 1
    most_likely_disease_label = plant_classes[most_likely_disease_idx]

    # Decide Healthy vs Diseased
    if diseased_prob > healthy_prob:
        plant_overall_label = "Diseased"
        # Aphid model input and prediction
        aphid_img = img.resize(aphid_size)
        aphid_array = np.array(aphid_img) / 255.0
        aphid_array = np.expand_dims(aphid_array, axis=0)

        aphid_probs = aphid_model.predict(aphid_array)[0]
        aphid_pred_idx = np.argmax(aphid_probs)
        aphid_pred_label = aphid_classes[aphid_pred_idx]
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

# Example usage
image_path = "/Users/sharyudilipmagre/Desktop/Example_data/exp1.jpg"
result = predict_leaf(image_path)

# Print results
print(f"Prediction for {image_path}:")
print(f"  \nOverall Plant Status: {result['plant_overall']}")
print(f"\n Healthy probability: {result['healthy_prob']*100:.2f}%")
print(f" Diseased probability: {result['diseased_prob']*100:.2f}%")
if result['plant_overall'] == "Diseased":
    print(f"  \nMost likely disease: {result['most_likely_disease']}")
print(" \nDetailed class probabilities:")
for cls, prob in result['plant_class_probs'].items():
    print(f"    {cls}: {prob*100:.2f}%")

# Only print aphid info if Diseased
if result['plant_overall'] == "Diseased":
    print(f"\nAphid Prediction: {result['aphid_class']}")
    print("\nAphid class probabilities:")
    for cls, prob in result['aphid_class_probs'].items():
        print(f"    {cls}: {prob*100:.2f}%")
