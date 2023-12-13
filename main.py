from flask import Flask, request
from ultralytics import YOLO
import io
from PIL import Image
import json

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("yolov8m.pt")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    image = Image.open(file)
    image.save("temp.jpg")

    # Perform object detection
    results = model("temp.jpg")
    # Convert the results to a Python dictionary
    results_dict = {}
    result = results[0]
    results_dict["Num_objs"] = len(result.boxes)
    results_dict["Objects"] = []
    for i in range(len(result.boxes)):
        box = result.boxes[i]
        cords = box.xyxy[0].tolist()
        class_id = result.names[box.cls[0].item()]
        conf = box.conf[0].item()
        l = {"Coordinated": cords, "Object_type": class_id, "Probabilty": conf}
        results_dict["Objects"].append(l)
    # Convert the dictionary to a JSON string
    results_json = json.dumps(results_dict)

    # Return the detected objects
    return results_json


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
