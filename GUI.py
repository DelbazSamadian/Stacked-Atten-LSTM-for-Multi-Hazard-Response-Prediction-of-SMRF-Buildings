import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 1️⃣ App Header
st.image(r"C:\Users\User\Downloads\OIP.jpeg", width=150)
st.markdown("""
# MIDR Prediction App  
Developed by **Teesside University**  
**Developers:** Delbaz Samadian, Imrose B. Muhit, Annalisa Occhipinti, Nashwan Dawood
""")

# 2️⃣ Load Means and Stds from the raw dataset
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

# 4️⃣ Load the trained model
device = torch.device("cpu")
static_input_size = 49
model = AttLSTM(input_size=2, static_input_size=static_input_size)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 5️⃣ Streamlit App Body
st.header("1️⃣ Upload Ground Motion Time Histories")
file_sa1 = st.file_uploader("Upload Sa1 Component (.txt)", type="txt")
file_sa2 = st.file_uploader("Upload Sa2 Component (.txt)", type="txt")
dt = st.number_input("Enter Time Step (dt in sec)", value=0.005, step=0.001, format="%.3f")

def process_time_history(file, dt):
    try:
        data = np.loadtxt(file)
        if data.ndim == 1:
            data_flat = data
        else:
            data_flat = data.flatten(order='C')  # row-wise flattening
        time = np.arange(0, len(data_flat) * dt, dt)
        return time, data_flat
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

def plot_time_history(time, sa_array, title):
    fig, ax = plt.subplots()
    ax.plot(time, sa_array)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Sa (g)")
    ax.set_title(title)
    st.pyplot(fig)

time_series_data1 = None
time_series_data2 = None

if file_sa1 is not None:
    time1, time_series_data1 = process_time_history(file_sa1, dt)
    if time1 is not None:
        plot_time_history(time1, time_series_data1, "Sa1 Time History")

if file_sa2 is not None:
    time2, time_series_data2 = process_time_history(file_sa2, dt)
    if time2 is not None:
        plot_time_history(time2, time_series_data2, "Sa2 Time History")

# 6️⃣ Ask user for Top 10 features (with explanations)
top_features = {
    'T1': 'Fundamental Period',
    'FH': 'Flood Height',
    'ϴp_Beam1': 'Plastic Rotation at Beam 1',
    'Ibeam1': 'Moment of Inertia at Beam 1',
    'M1': 'Mass at Floor 1',
    'ϴpc_Col_Ex1': 'Plastic Hinge Rotation at Exterior Column 1',
    'ϴp_Col_Ex1': 'Plastic Rotation at Exterior Column 1',
    'Abeam1': 'Area of Beam 1',
    'ϴpc_Col_In1': 'Plastic Hinge Rotation at Interior Column 1',
    'someFeature10': 'Additional Feature 10'
}

st.header("2️⃣ Enter Top 10 Most Important Features (Raw Values)")
user_inputs = {}
for feature, description in top_features.items():
    default_val = float(feature_means.get(feature, 0.0))
    user_inputs[feature] = st.number_input(f"{feature} ({description}):", value=default_val)

# 7️⃣ Predict MIDR
if st.button("Predict MIDR"):
    if time_series_data1 is not None and time_series_data2 is not None:
        def normalize(ts):
            return (ts - ts.mean()) / ts.std() if ts.std() > 0 else ts - ts.mean()
        ts1 = normalize(time_series_data1)
        ts2 = normalize(time_series_data2)

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
            raw_value = user_inputs[col] if col in user_inputs else feature_means.get(col, 0.0)
            mean = feature_means.get(col, 0.0)
            std = feature_stds.get(col, 1.0)
            standardized_value = (raw_value - mean) / std
            static_inputs.append(standardized_value)
        static_tensor = torch.tensor([static_inputs], dtype=torch.float32).to(device)

        with torch.no_grad():
            midr = model(x_time_series, static_tensor).cpu().item()
        st.success(f"Predicted MIDR: {midr:.6f}")
    else:
        st.warning("Please upload both Sa1 and Sa2 files and enter dt before predicting.")
