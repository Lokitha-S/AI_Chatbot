<img width="1470" height="829" alt="Screenshot 2025-11-19 at 11 50 30 AM" src="https://github.com/user-attachments/assets/439b9496-be21-45a7-926e-16b40036dd6b" />

# 🩺 AI Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## 📌 Overview
The AI Disease Prediction System is a machine learning-based healthcare application designed to assist users in identifying potential diseases based on their symptoms. 

Built with Python and Streamlit, the system utilizes a Random Forest Classifier to analyze user inputs against a dataset of over 40 diseases and 130+ symptoms. Beyond simple prediction, it acts as a virtual health consultant by providing disease descriptions, severity warnings, and precautionary measures.

## 🚀 Features
* **Symptom-Based Diagnosis:** Users can select multiple symptoms from a drop-down list.
* **Machine Learning Model:** Uses a Random Forest Classifier for high-accuracy multi-class classification.
* **Confidence Score:** Displays the probability percentage of the diagnosis.
* **Comprehensive Reports:** Provides the disease description and 4 specific precautions/remedies.
* **Severity Alerts:** Flags high-risk conditions that require immediate medical attention.
* **User-Friendly UI:** A clean, responsive web interface built with Streamlit.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **ML Libraries:** Scikit-Learn, NumPy, Pandas
* **Data Source:** Custom Healthcare CSV Datasets (Training & Master Data)

##Disclaimer
This project is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition.




## 📂 Project Structure
```text
├── app.py                   # Main Streamlit application file
├── requirements.txt         # List of dependencies
├── Data/
│   ├── Training.csv         # Dataset for training the model
│   └── Testing.csv          # Dataset for validation
└── MasterData/
    ├── symptom_Description.csv  # Disease descriptions
    ├── symptom_precaution.csv   # Precautionary measures
    └── symptom_severity.csv     # Severity ratings

