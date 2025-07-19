# Arduino SineWave LED with TensorFlow Lite

This project demonstrates how to use a **quantized TensorFlow Lite model** running on an **Arduino microcontroller** to generate a **sine wave approximation**, which controls the **brightness of an LED** to simulate a smooth pulsing or breathing light effect.


## How It's Made

### Tech Used
- **TensorFlow / TensorFlow Lite**
- **Keras**
- **Python (for model training & quantization)**
- **C++ (Arduino programming)**
- **Arduino IDE / PlatformIO**

### Key Features
- **Neural Network Model:** Trained using Keras to approximate the sine function \( y = \sin(x) \) with added noise for robustness.
- **Model Quantization:** Converted to a **quantized `.tflite` model** for efficient inference on microcontrollers.
- **TFLite Micro Interpreter:** Runs the trained model on the Arduino.
- **LED Brightness Control:** The model's output controls the LED brightness, creating a smooth sine wave lighting effect.

---

## Key Components

### Python Model Training
- **Data Generation:** Simulates noisy sine wave data.
- **Model Architecture:** A feedforward neural network with multiple hidden layers.
- **Quantization:** Optimizes the model for embedded hardware.

### Embedded C++ Code
- **`setup()`**
  - Initializes the TensorFlow Lite interpreter and loads the quantized model.
  - Allocates memory for tensors.
- **`loop()`**
  - Feeds incremented \( x \) values to the model.
  - Captures the output \( y \approx \sin(x) \).
  - Sends \( y \) to `HandleOutput` to control the LED.

### `HandleOutput(x, y)`
- Maps \( y \) from the range `[-1, 1]` to `[0, 255]` for PWM output.
- Adjusts LED brightness accordingly.

---

## Optimizations

- **Model Quantization:** Reduces memory usage and inference time.
- **Efficient Memory Usage:** Pre-allocated tensor arena ensures predictable performance.
- **Minimal CPU Load:** Lightweight model enables real-time processing on low-power devices.

---

## Lessons Learned

- **Embedded Machine Learning:** Gained experience in deploying ML models on microcontrollers with TensorFlow Lite Micro.
- **Quantization Techniques:** Learned how to optimize models for size and speed using post-training quantization.
- **C++ and Embedded Programming:** Improved skills in efficient programming for constrained devices.
- **Signal Processing:** Applied neural networks for approximating mathematical functions.

---

## Future Improvements

- Control multiple LEDs for multi-phase sine waves.
- Extend the model to approximate other waveforms (cosine, triangle, sawtooth).
- Add user controls for wave parameters using sensors or buttons.
