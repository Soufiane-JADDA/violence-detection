# 🚨 Violence Detection in Videos using CNN + LSTM

This project implements a real-time violence detection system using deep learning. It combines **ResNet18** (for spatial feature extraction) with **LSTM** (for temporal modeling), trained on video clips to classify scenes as *violent* or *non-violent*.

---

## 🧠 Model Architecture

- **Backbone**: ResNet18 (pretrained, without final FC layer)
- **Temporal**: LSTM with 128 hidden units
- **Classifier**: Fully connected layer
- **Input**: Sequence of RGB video frames (20 frames per clip)

---

## 📁 Project Structure

```

violence-detection/
├── model/                   # Model definition (CNN + LSTM)
├── data/                    # Dataset directory (train/test)
├── utils                    # Helper functions (frame extraction, etc.)
├── checkpoints/             # Saved models
├── train.py                 # Main training script
├── live\_detection.py       # Real-time webcam inference
├── evaluate.py              # Evaluation & classification report
├── requirements.txt         # Required Python packages
└── README.md

````

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Soufiane-JADDA/violence-detection.git
cd violence-detection

# Create a virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🏋️‍♂️ Training the Model

```bash
python train.py
```

The model will:

* Train using the Real-Life Violence Dataset
* Save checkpoints and the best model (`violence_detector_best.pt`)
* Plot training and validation loss over epochs

---

## 🎥 Real-Time Detection (Webcam)

```bash
python live_detection.py
```

You will see video frames and detection status:

* 🟢 **NON-VIOLENT**
* 🚨 **VIOLENT** (if probability > 80%)

---

## 📊 Evaluation

Run classification report and confusion matrix:

```bash
python evaluate.py
```

You’ll get:

* Precision / Recall / F1-score
* Confusion matrix
* Misclassified samples saved for analysis

---

## 🧪 Sample Output

```
Classification Report:
              precision    recall  f1-score   support
 Non-Violent       0.98      0.92      0.95      1000
     Violent       0.92      0.98      0.95      1000
 Accuracy                              0.95      2000
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the project
2. Create your feature branch: `git checkout -b feat/feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feat/feature-name`
5. Open a pull request

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♀️ Maintainer

**Soufiane Jadda**

* GitHub: [@Soufiane-JADDA](https://github.com/Soufiane-JADDA)
* Email: [soufiane.jadda@usmba.ac.ma](mailto:soufiane.jadda@usmba.ac.ma)

```
 
