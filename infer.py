import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("Models/ANN.onnx")

def predict(inputs):
    inputs = np.array(inputs, dtype=np.float32).reshape(1, -1)
    outputs = session.run(["output"], {"input": inputs})

    logits = outputs[0][0]
    prob = 1 / (1 + np.exp(-logits))  # sigmoid
    prediction = (prob >= 0.5).astype(int)

    return int(prediction[0])

if __name__ == '__main__':
    inputs = [3, 1, 22.0, 1, 0, 7.25, 2]

    prediction = predict(inputs)

    print(prediction)