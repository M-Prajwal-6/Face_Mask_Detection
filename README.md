# Face Mask Detection

This project implements a face mask detection system using a deep learning model.

## Features
- Detects whether a person is wearing a face mask
- Includes a Flask web application for inference
- Contains training code and a pre-trained model file

## Project Files
- `app.py` - Flask application for running the detector
- `train_model.py` - Training script for the model
- `face_mask_model.h5` - Pre-trained model weights
- `requirements.txt` - Python dependencies

## Setup
1. Create a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```

## Notes
- The model file is large and may require sufficient disk space.
- For best results, run the application in a Python environment with TensorFlow installed.
