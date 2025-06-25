# Scam Detection Using NLP & Machine Learning

## Overview

This project is a Scam Detection System designed to identify fraudulent messages across text-based communication platforms like (e.g., WhatsApp, Instagram, Telegram). It uses a hybrid approach combining Natural Language Processing (NLP) techniques (TF-IDF, N-gram analysis) and Machine Learning (Random Forest) to detect scams with high accuracy and minimal false positives.

## Features

- **Hybrid Detection:** Combines pattern-based (N-gram) and ML-based (TF-IDF + Random Forest) analysis.
- **Multi-Platform:** Detects scams from various messaging platforms.
- **Scam Type Classification:** Identifies the type of scam (phishing, impersonation, financial, urgent, etc.).
- **Continuous Learning:** Learns from user feedback to adapt to new scam tactics.
- **Educational Component:** Provides information to help users recognize scams.

## Project Structure

- `run.py` — Main entry point; starts the Flask web server and runs evaluation.
- `scam_detector.py` — Core logic for text processing, feature extraction, and ML models.
- `routes.py` — Flask routes for web/API interaction.
- `index.html` — Web interface for user interaction.
- `__init__.py` — Flask app initialization.
- `Financial.html` — Educational page for financial scams.
- `Phishing.html` — Educational page for phishing scams.
- `Urgency.html` — Educational page for urgent scams.
- `Impersonation.html` — Educational page for impersonation scams.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Scam-Detection-Using-NLP-Machine-Learning-Text-Based.git
   cd Scam-Detection-Using-NLP-Machine-Learning-Text-Based
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install flask scikit-learn pandas numpy nltk
   ```

4. **Download NLTK resources (automatically handled on first run):**
   - The system will download required NLTK resources (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`) if not already present.

## Usage

1. **Run the application:**
   ```bash
   python run.py
   ```

2. **Access the web interface:**
   - Open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **How it works:**
   - Paste or type a message into the web interface.
   - The system will analyze the message and display the risk level, scam type, and important features.
   - You can provide feedback to help the system learn and improve.

## Evaluation

- On startup, the system performs a comprehensive evaluation (cross-validation and holdout test) and prints results in the terminal.
- Metrics include accuracy, precision, recall, F1 score, and a confusion matrix.

## Notes

- This is a Web Application (Not directly implemented into WhatsApp or Instagram.
- The system currently supports only English-language messages.
- Training data is stored in `data/training_data.csv` and grows as users provide feedback.
- In the event of dataset corruption, the system incorporates a backup mechanism that restores the dataset using the flagged_csv file. (This feature was implemented to address issues encountered during the production phase of the project)

## License

MIT License
