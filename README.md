# Stacked Attention LSTM for Multi-Hazard Response Prediction of SMRF Buildings

Welcome to the **Stacked Attention LSTM for Multi-Hazard Response Prediction of SMRF Buildings** repository! This project provides a user-friendly web application built with Streamlit that predicts the Multi-Hazard Immediate Seismic Response (MISR) of SMRF (Special Moment Resisting Frame) buildings. This can help engineers and researchers understand how structures perform under multiple hazards.

# Features
Predicts MISR given input parameters.  
Easy-to-use, interactive web interface with Streamlit.  
Open-source, free to use and customize.  
---

## Live Demo

👉 [Launch the Streamlit App](https://stack-atten-lstm-sm.streamlit.app/)  


## How to Use

1️⃣ **Input Building and Hazard Parameters**  
- Enter time history of Sa in two directions,
- Enter other features like flood height, T1, beam and column properties, etc.

2️⃣ **Run Prediction**  
- Click the **Predict MIDR** button to compute the MISR using the stacked attention LSTM model.

3️⃣ **View Results**  
- The app displays the predicted MISR for the given inputs.



## 📥 Getting Started Locally

Follow these steps to run the app on your own machine:

   ## 1. Clone the Repository

```bash
git clone https://github.com/DelbazSamadian/Stacked-Atten-LSTM-for-Multi-Hazard-Response-Prediction-of-SMRF-Buildings.git
cd Stacked-Atten-LSTM-for-Multi-Hazard-Response-Prediction-of-SMRF-Buildings

  ## 2. (Optional but Recommended) Create a Virtual Environment
# Windows:
python -m venv venv
venv\Scripts\activate

# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

  ## 3. Install Dependencies
Make sure you have Python 3.8+ installed, then:
pip install -r requirements.txt

  ## 4. Run the App
streamlit run GUI.py


