import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 0️⃣ Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .block-container {
        padding-top: 5px;
        padding-bottom: 5px;
    }
    .stFileUploader {
        padding: 2px !important;
        margin-bottom: 4px !important;
        height: 40px !important;
    }
    .uploadedFile {
        margin-bottom: 4px !important;
    }
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

# 1️⃣ App Header
st.image("OIP.jpeg", width=150, use_column_width=False)
st.markdown("""
# MIDR Prediction App  
Developed by **Teesside University**  
**Developers:** Delbaz Samadian, Imrose B. Muhit, Annalisa Occhipinti, Nashwan Dawood
""")

# 2️⃣ Load Means and Stds (same)
raw_data_path = "filtered_refinedDoE1.xlsx"
df_raw = pd.read_excel(raw_data_path)
non_input_cols = ["iModel", "iRP", "irecord", "MDR", "MFA", "MBS"]
input_cols = [col for col in df_raw.columns if col not in non_input_cols]
feature_means = df_raw[input_cols].mean().to_dict()
feature_stds = df_raw[input_cols].std().replace(0, 1e-6).to_dict()

# 3️⃣ Model Definitions (same)
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

# 4️⃣ Load model (same)
device = torch.device("cpu")
model = AttLSTM(input_size=2, static_input_size=49)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 5️⃣ Three-Column Layout
col1, col2, col3 = st.columns([1.2, 1.5, 1.2])

# Sa Uploads & Plots
with col1:
    st.subheader("Sa Input")
    file_sa1 = st.file_uploader("Sa1 (.txt)", type="txt", label_visibility="collapsed")
    file_sa2 = st.file_uploader("Sa2 (.txt)", type="txt", label_visibility="collapsed")
    dt = st.number_input("dt (sec)", value=0.005, step=0.001)
    def process_time(file, dt):
        try:
            data = np.loadtxt(file)
            data_flat = data.flatten(order='C')
            time = np.arange(0, len(data_flat) * dt, dt)
            return time, data_flat
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None, None
    def plot_history(time, data, label):
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(time, data)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Sa (g)")
        ax.set_title(label)
        st.pyplot(fig)
    time1, time2 = None, None
    if file_sa1:
        time1, data1 = process_time(file_sa1, dt)
        if time1 is not None: plot_history(time1, data1, "Sa1 Time History")
    if file_sa2:
        time2, data2 = process_time(file_sa2, dt)
        if time2 is not None: plot_history(time2, data2, "Sa2 Time History")

# Features
with col2:
    st.subheader("Top 10 Features")
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
    keys = list(top_features.keys())
    for i in range(0, len(keys), 2):
        row = st.columns(2)
        for j in range(2):
            if i + j < len(keys):
                key = keys[i + j]
                desc = top_features[key]
                default_val = feature_means.get(key, 0.0)
                user_inputs[key] = row[j].number_input(f"{key} ({desc}):", value=default_val)

# Prediction
with col3:
    st.subheader("MIDR Prediction")
    if st.button("Predict MIDR"):
        if time1 is not None and time2 is not None:
            def norm(ts): return (ts - ts.mean()) / ts.std() if ts.std() > 0 else ts - ts.mean()
            ts1, ts2 = norm(data1), norm(data2)
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
