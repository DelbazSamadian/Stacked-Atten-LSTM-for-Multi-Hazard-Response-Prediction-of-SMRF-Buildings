import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 1️⃣ Load means and stds from the raw dataset
raw_data_path = "filtered_refinedDoE1.xlsx"
df_raw = pd.read_excel(raw_data_path)
non_input_cols = ["iModel", "iRP", "irecord", "MDR", "MFA", "MBS"]
input_cols = [col for col in df_raw.columns if col not in non_input_cols]
feature_means = df_raw[input_cols].mean().to_dict()
feature_stds = df_raw[input_cols].std().replace(0, 1e-6).to_dict()  # prevent division by zero

# 2️⃣ Model Definitions
class Attention(torch.nn.Module):
    def __init__(self, hidden_size, static_input_size):
        super(Attention, self).__init__()
        self.Wq = torch.nn.Linear(hidden_size, hidden_size)
        self.Wk = torch.nn.Linear(hidden_size, hidden_size)
        self.tanh = torch.tanh
        self.static_fc = torch.nn.Linear(static_input_size, hidden_size)

    def forward(self, hidden_states, static_inputs):
        Q = torch.tanh(self.Wq(hidden_states[:, -1, :]))
        K = torch.tanh(self.Wk(hidden_states))
        attention_scores = torch.bmm(K, Q.unsqueeze(2)).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(hidden_states.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        static_features = self.static_fc(static_inputs)
        context_vector = torch.cat((context_vector, static_features), dim=1)
        return context_vector

class AttLSTM(torch.nn.Module):
    def __init__(self, input_size, static_input_size, hidden_size=64, num_layers=2, dropout_rate=0.2, fc_dropout_rate=0.3):
        super(AttLSTM, self).__init__()
        effective_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm_encoder = torch.nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=effective_dropout)
        self.attention = Attention(hidden_size, static_input_size)
        self.lstm_decoder = torch.nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc1 = torch.nn.Linear(hidden_size, 64)
        self.fc_dropout = torch.nn.Dropout(fc_dropout_rate)
        self.fc2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, static_input):
        if x.dim() == 4:
            batch_size, num_stacks, stack_len, input_size = x.size()
            x = x.view(batch_size, num_stacks * stack_len, input_size)
        encoder_output, _ = self.lstm_encoder(x)
        attention_output = self.attention(encoder_output, static_input)
        decoder_output, _ = self.lstm_decoder(attention_output.unsqueeze(1))
        final_output = self.fc1(decoder_output[:, -1, :])
        final_output = self.relu(final_output)
        final_output = self.fc_dropout(final_output)
        return self.fc2(final_output).squeeze(-1)

# 3️⃣ Load the trained model
device = torch.device("cpu")
static_input_size = 49  # trained model expects all 49 features
model = AttLSTM(input_size=2, static_input_size=static_input_size)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 4️⃣ Streamlit App
st.set_page_config(page_title="MIDR Prediction App", layout="wide")

# Teesside University Branding
col1, col2 = st.columns([1, 5])
with col1:
    st.image("path/to/tees_logo.png", width=150)  # Replace with correct path
with col2:
    st.markdown("""
    # MIDR Prediction App  
    **Developed by Teesside University**  
    **Developers**: Delbaz Samadian, Imrose B. Muhit, Annalisa Occhipinti, Nashwan Dawood
    """)

st.markdown("---")

# 5️⃣ Upload Section
st.header("📂 Upload Ground Motion Time Histories")
file_sa1 = st.file_uploader("Upload Sa1 Component (.txt)", type="txt")
file_sa2 = st.file_uploader("Upload Sa2 Component (.txt)", type="txt")

time_series_data1 = None
time_series_data2 = None

if file_sa1 is not None:
    try:
        time_series_data1 = np.loadtxt(file_sa1)
        st.success("✅ Sa1 loaded successfully!")
        fig, ax = plt.subplots()
        time_axis = np.arange(len(time_series_data1))
        ax.plot(time_axis, time_series_data1)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Sa (g)")
        ax.set_title("Sa1 Time History")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error reading Sa1 file: {e}")

if file_sa2 is not None:
    try:
        time_series_data2 = np.loadtxt(file_sa2)
        st.success("✅ Sa2 loaded successfully!")
        fig, ax = plt.subplots()
        time_axis = np.arange(len(time_series_data2))
        ax.plot(time_axis, time_series_data2)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Sa (g)")
        ax.set_title("Sa2 Time History")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error reading Sa2 file: {e}")

st.markdown("---")

# 6️⃣ Feature Input Section
top_features = ['T1', 'FH', 'ϴp_Beam1', 'Ibeam1', 'M1', 'ϴpc_Col_Ex1', 'ϴp_Col_Ex1', 'Abeam1', 'ϴpc_Col_In1', 'someFeature10']

st.header("📝 Enter Top 10 Most Important Features (Raw Values)")

with st.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
    - **T1**: Fundamental period (sec)
    - **FH**: Flood height (m)
    - **ϴp_Beam1**: Plastic rotation of the first beam (rad)
    - **Ibeam1**: Moment of inertia of the first beam (cm⁴)
    - **M1**: Mass of the first floor (tons)
    - **ϴpc_Col_Ex1**: Plastic rotation capacity of exterior column (rad)
    - **ϴp_Col_Ex1**: Plastic rotation of exterior column (rad)
    - **Abeam1**: Cross-sectional area of the first beam (cm²)
    - **ϴpc_Col_In1**: Plastic rotation capacity of interior column (rad)
    - **someFeature10**: Placeholder feature (customizable)
    """)

user_inputs = {}
cols = st.columns(5)
for i, feature in enumerate(top_features):
    with cols[i % 5]:
        default_val = float(feature_means.get(feature, 0.0))
        value = st.number_input(f"{feature}:", value=default_val)
        user_inputs[feature] = value

st.markdown("---")

# 7️⃣ Prediction Section
if st.button("🚀 Predict MIDR"):
    if time_series_data1 is not None and time_series_data2 is not None:
        ts1 = time_series_data1.flatten()
        ts2 = time_series_data2.flatten()

        def normalize(ts):
            return (ts - ts.mean()) / ts.std() if ts.std() > 0 else ts - ts.mean()

        ts1 = normalize(ts1)
        ts2 = normalize(ts2)

        TIME_STEPS = 9000
        pad_len = TIME_STEPS - min(len(ts1), len(ts2))
        ts1 = np.concatenate([ts1, np.full(pad_len, ts1.mean())]) if pad_len > 0 else ts1[:TIME_STEPS]
        ts2 = np.concatenate([ts2, np.full(pad_len, ts2.mean())]) if pad_len > 0 else ts2[:TIME_STEPS]

        ts_data = np.stack((ts1, ts2), axis=1)

        STACK_SIZE = 120
        OVERLAP = 0.5
        step = int(STACK_SIZE * (1 - OVERLAP))
        stacked_input = []
        for i in range(0, TIME_STEPS - STACK_SIZE + 1, step):
            window = ts_data[i:i + STACK_SIZE]
            stacked_input.append(torch.tensor(window, dtype=torch.float32))
        x_time_series = torch.stack(stacked_input).unsqueeze(0).to(device)

        static_inputs = []
        for col in input_cols:
            if col in user_inputs:
                raw_value = user_inputs[col]
            else:
                raw_value = feature_means.get(col, 0.0)
            mean = feature_means.get(col, 0.0)
            std = feature_stds.get(col, 1.0)
            standardized_value = (raw_value - mean) / std
            static_inputs.append(standardized_value)

        static_tensor = torch.tensor([static_inputs], dtype=torch.float32).to(device)

        with torch.no_grad():
            midr = model(x_time_series, static_tensor).cpu().item()
        st.success(f"🎯 **Predicted MIDR:** {midr:.6f}")
    else:
        st.warning("⚠️ Please upload both Sa1 and Sa2 files before predicting.")

# --- END OF APP ---
