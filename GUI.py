import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 0️⃣ Custom CSS for fonts and styling
st.markdown("""
    <style>
    body, .stApp, .block-container {
        font-family: 'Times New Roman', Times, serif;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .block-container {
        padding-top: 5px;
        padding-bottom: 5px;
    }
    .stFileUploader {
        background-color: #ddd !important;
        border-radius: 5px !important;
        border: 1px solid #888 !important;
        padding: 8px !important;
    }
    .stNumberInput input {
        background-color: #ddd !important;
        border-radius: 5px !important;
        padding: 6px !important;
        font-family: 'Times New Roman', Times, serif;
    }
    .stButton>button {
        background-color: #444 !important;
        color: white !important;
        border-radius: 5px !important;
        font-family: 'Times New Roman', Times, serif;
    }
    .stMarkdown, .stSubheader, .stHeading, .stSuccess, .stWarning {
        font-family: 'Times New Roman', Times, serif;
    }
    .stTextInput label, .stNumberInput label {
        font-family: 'Times New Roman', Times, serif;
    }
    </style>
""", unsafe_allow_html=True)

# 1️⃣ App Header
st.markdown("""
# MIDR Prediction App  
Developed by **Teesside University**  
**Developers:** Delbaz Samadian, Imrose B. Muhit, Annalisa Occhipinti, Nashwan Dawood
""")

# 2️⃣ Load Means and Stds
raw_data_path = "filtered_refinedDoE1.xlsx"
df_raw = pd.read_excel(raw_data_path)
non_input_cols = ["iModel", "iRP", "irecord", "MDR", "MFA", "MBS"]
input_cols = [col for col in df_raw.columns if col not in non_input_cols]
feature_means = df_raw[input_cols].mean().to_dict()
feature_stds = df_raw[input_cols].std().replace(0, 1e-6).to_dict()

# 3️⃣ Model Definitions
class Attention(torch.nn.Module):
    def __init__(self, hidden_size, static_input_size):
        super(Attention, self).__init__()
        self.Wq = torch.nn.Linear(hidden_size, hidden_size)
        self.Wk = torch.nn.Linear(hidden_size, hidden_size)
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

# 4️⃣ Load model
device = torch.device("cpu")
model = AttLSTM(input_size=2, static_input_size=49)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 🟠 Horizontal layout: Sa1 and Sa2 uploaders and time histories side by side
cols_top = st.columns([1, 2, 1, 2])  # uploader, plot, uploader, plot

# Sa1 Upload and Plot
with cols_top[0]:
    dt = st.number_input("Time Step (dt in sec)", value=0.005, step=0.001)
    file_sa1 = st.file_uploader("Sa1 Component (.txt)", type="txt", key="sa1")
with cols_top[1]:
    if file_sa1:
        try:
            data1 = np.loadtxt(file_sa1).flatten(order='C')
            time1 = np.arange(0, len(data1) * dt, dt)
            fig1, ax1 = plt.subplots(figsize=(6, 2.5))
            ax1.plot(time1, data1, color='b')
            ax1.set_xlabel("Time (sec)")
            ax1.set_ylabel("Sa (g)")
            ax1.set_title("Sa1 Time History")
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"Error reading Sa1: {e}")

# Sa2 Upload and Plot
with cols_top[2]:
    file_sa2 = st.file_uploader("Sa2 Component (.txt)", type="txt", key="sa2")
with cols_top[3]:
    if file_sa2:
        try:
            data2 = np.loadtxt(file_sa2).flatten(order='C')
            time2 = np.arange(0, len(data2) * dt, dt)
            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
            ax2.plot(time2, data2, color='r')
            ax2.set_xlabel("Time (sec)")
            ax2.set_ylabel("Sa (g)")
            ax2.set_title("Sa2 Time History")
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error reading Sa2: {e}")


# 🟠 Columns 3 and 4: Top 10 Features
top_features = {
    'T1': 'Fundamental Period',
    'FH': 'Flood Height',
    'ϴp_Beam1': 'Pre-cap rot at Beam 1',
    'Ibeam1': 'Moment of Inertia Beam 1',
    'M1': 'First Mode Mass Part.',
    'ϴpc_Col_Ex1': 'Post-cap rot Ext Col 1',
    'ϴp_Col_Ex1': 'Pre-cap rot Ext Col 1',
    'Abeam1': 'Area Beam 1',
    'ϴpc_Col_In1': 'Post-cap rot Int Col 1',
    'someFeature10': 'Additional Feature 10'
}
user_inputs = {}
feature_keys = list(top_features.keys())
for idx, key in enumerate(feature_keys):
    col = cols[2] if idx < 5 else cols[3]
    desc = top_features[key]
    default_val = feature_means.get(key, 0.0)
    user_inputs[key] = col.number_input(f"{key} ({desc}):", value=default_val)

# 🟠 Column 5: MIDR Prediction
with cols[4]:
    st.subheader("MIDR Prediction")
    if st.button("Predict MIDR"):
        if file_sa1 and file_sa2:
            def norm(ts): return (ts - ts.mean()) / ts.std() if ts.std() > 0 else ts - ts.mean()
            ts1 = norm(data1)
            ts2 = norm(data2)
            TIME_STEPS = 9000
            pad_len = 9000 - min(len(ts1), len(ts2))
            ts1 = np.concatenate([ts1, np.full(pad_len, ts1.mean())]) if pad_len > 0 else ts1[:TIME_STEPS]
            ts2 = np.concatenate([ts2, np.full(pad_len, ts2.mean())]) if pad_len > 0 else ts2[:TIME_STEPS]
            ts_data = np.stack((ts1, ts2), axis=1)
            step = int(120 * (1 - 0.5))
            stacked_input = [torch.tensor(ts_data[i:i+120], dtype=torch.float32)
                             for i in range(0, TIME_STEPS - 120 + 1, step)]
            x_ts = torch.stack(stacked_input).unsqueeze(0)
            static_inputs = []
            for col in input_cols:
                val = user_inputs.get(col, feature_means.get(col, 0.0))
                std = feature_stds.get(col, 1.0)
                mean = feature_means.get(col, 0.0)
                static_inputs.append((val - mean) / std)
            x_static = torch.tensor([static_inputs], dtype=torch.float32)
            with torch.no_grad():
                midr = model(x_ts, x_static).cpu().item()
            st.success(f"Predicted MIDR: {midr:.6f}")
        else:
            st.warning("Upload both Sa1 & Sa2 and enter dt.")
