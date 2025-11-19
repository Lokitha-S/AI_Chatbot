import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for a clean UI
warnings.filterwarnings("ignore")

# --- Configuration ---
st.set_page_config(page_title="HealthCare AI", page_icon="🩺", layout="wide")

# --- Logic Class ---
class DiseasePredictor:
    def __init__(self):
        self.DATA_PATH = 'Data/Training.csv'
        self.DESC_PATH = 'MasterData/symptom_Description.csv'
        self.PRECAUTION_PATH = 'MasterData/symptom_precaution.csv'
        self.SEVERITY_PATH = 'MasterData/symptom_severity.csv'
        
        # Load data immediately
        self.train_df = self._load_training_data()
        self.description_dict, self.precaution_dict, self.severity_dict = self._load_knowledge_base()
        
    def _load_training_data(self):
        try:
            df = pd.read_csv(self.DATA_PATH)
            # Clean column names
            df.columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
            df = df.loc[:, ~df.columns.duplicated()]
            return df
        except FileNotFoundError:
            st.error("❌ Critical Error: 'Data/Training.csv' not found!")
            st.stop()

    def _load_knowledge_base(self):
        try:
            desc_df = pd.read_csv(self.DESC_PATH)
            desc_dict = dict(zip(desc_df.iloc[:, 0], desc_df.iloc[:, 1]))

            prec_df = pd.read_csv(self.PRECAUTION_PATH)
            prec_dict = {row[0]: list(row[1:]) for row in prec_df.values}

            sev_df = pd.read_csv(self.SEVERITY_PATH)
            sev_dict = dict(zip(sev_df.iloc[:, 0], sev_df.iloc[:, 1]))
            return desc_dict, prec_dict, sev_dict
        except FileNotFoundError:
            st.warning("⚠️ Knowledge base files missing. Descriptions will be unavailable.")
            return {}, {}, {}

# --- Caching the Model (Crucial for Streamlit) ---
@st.cache_resource
def train_model(df):
    # Prepare data
    all_symptoms = df.columns[:-1]
    X = df[all_symptoms]
    y = df['prognosis']
    
    # Encode
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le, all_symptoms

# --- UI Layout ---
def main():
    # Initialize Class
    predictor = DiseasePredictor()
    model, le, all_symptoms = train_model(predictor.train_df)

    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.sidebar.title("HealthCare AI")
    st.sidebar.info("Select your symptoms from the dropdown to get a prediction.")

    # Main Content
    st.title("🩺 AI Disease Prediction System")
    st.markdown("---")

    # Symptom Selection
    # Formatting symptoms for display (removing underscores)
    symptom_options = list(all_symptoms)
    selected_symptoms = st.multiselect(
        "Select your Symptoms:", 
        options=symptom_options,
        format_func=lambda x: x.replace("_", " ").title()
    )

    if st.button("Analyze Symptoms", type="primary"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            # Prepare Input Vector
            input_vector = np.zeros(len(all_symptoms))
            symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
            
            for symptom in selected_symptoms:
                if symptom in symptom_index:
                    input_vector[symptom_index[symptom]] = 1

            # Prediction
            pred_proba = model.predict_proba([input_vector])[0]
            pred_idx = np.argmax(pred_proba)
            disease = le.inverse_transform([pred_idx])[0]
            confidence = round(pred_proba[pred_idx] * 100, 2)

            # Display Results
            st.success(f"**Diagnosis:** {disease}")
            st.progress(int(confidence))
            st.caption(f"Confidence Score: {confidence}%")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📖 Description")
                st.write(predictor.description_dict.get(disease, "No description available."))

            with col2:
                st.subheader("🛡️ Precautions")
                if disease in predictor.precaution_dict:
                    for i, p in enumerate(predictor.precaution_dict[disease], 1):
                        st.write(f"{i}. {p}")
                else:
                    st.write("No specific precautions found.")

            # Severity Warning
            severity = predictor.severity_dict.get(disease, 0)
            if severity > 5:
                st.error(f"⚠️ Warning: This condition has a high severity index ({severity}/7). Please consult a doctor immediately.")

if __name__ == "__main__":
    main()