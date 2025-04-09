# Car Shadow Detection Model

This project uses a YOLO model (in ONNX format) to detect various shadow types around cars, such as front, back, and side shadows. It also allows overlaying the car on a new background and draws realistic shadows based on the detection results.

---

## 📆 Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/t70256242/Car-Shadow-Detection-Model.git
cd Car-Shadow-Detection-Model
```

---

## 🚀 Usage

### 🔍 Model Inference

Use the trained YOLO model to detect shadows in a car image. Here’s an example of how to load the `.onnx` model and run inference on an image:

```python
import cv2
import torch
import onnx
import numpy as np

# Load the ONNX model
model = onnx.load('best.onnx')

# Load your image
image = cv2.imread('car_image.jpg')

# Preprocess and run the model (assuming you have a pre-trained model)
# Model inference steps here...
```

---

## 🏋️ Train Your Own Model

If you want to train the model yourself, ensure you have labeled data in YOLO format (with classes like `back_shadow`, `front_shadow`, `side_shadow`, and `whole_car`). Follow the YOLOv5 training steps below:

```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Install dependencies
pip install -U -r requirements.txt

# Train the model
python train.py --img-size 640 --batch-size 16 --epochs 50 --data custom_data.yaml --weights yolov5s.pt
```

---

## 🖼️ Example Outputs

The model detects shadow regions around the car:

- **Back Shadow**: Detected at the rear of the car.
- **Front Shadow**: Detected at the front of the car.
- **Side Shadow**: Detected along the sides of the car.
- **Whole Car Shadow**: Detection covering the full car body.

The model can be used to create realistic image compositions by placing the car on new backgrounds and drawing shadows in the correct orientation.

---

## 🤝 Contributing

Feel free to fork this repository and contribute by submitting issues or pull requests. If you have improvements or suggestions, don't hesitate to open an issue.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) for the base detection architecture.
- [OpenCV](https://opencv.org/) for image processing utilities.
- [ONNX](https://onnx.ai/) for model deployment.

---

## 📬 Contact

For any inquiries, feel free to contact **Ayo Ayorinde**.

