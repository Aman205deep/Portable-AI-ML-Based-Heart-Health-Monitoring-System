# ECG & Pulse Abnormality Detection using Machine Learning

A Random Forest–based project that classifies ECG/pulse data as normal or abnormal. Built for Smart India Hackathon.

# Disclaimer:
Educational project only — not a medical device, not a diagnosis tool.

# Why This Project
Reading ECG signals for irregularities usually needs a trained eye and takes time. We wanted to see how well a simple ML pipeline could flag potentially abnormal patterns automatically.

# Approach
ECG/pulse data → preprocessing → feature extraction → train/test split → Random Forest → prediction
We picked Random Forest because it combines multiple decision trees, handles tabular data well, resists overfitting better than a single tree, and gives us feature importance for free.

# Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib

# Project Structure
```
ECG-Abnormality-Detection/
├── dataset/dataset.csv
├── main.py
├── model.py
├── preprocessing.py
├── requirements.txt
└── README.md
```

# Running It
```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd ECG-Abnormality-Detection
pip install -r requirements.txt
python main.py
```

# Evaluation
Accuracy, precision, recall, F1-score, and confusion matrix — not just accuracy alone, since that can be misleading.

# Future Scope
Real ECG sensors, a mobile/web interface, real-time alerts, live waveform visualization, cloud storage, and comparing against deep learning models (CNN/LSTM).

# Team Enigmax
Built for Smart India Hackathon.

Disclaimer
Not a medical diagnosis. See a real doctor for actual health concerns.
