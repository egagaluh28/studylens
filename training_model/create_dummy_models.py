# create_dummy_models.py
from utils.cnn_model import build_cnn_model
from utils.lstm_model import build_lstm_model

# Buat CNN dummy
cnn_dummy = build_cnn_model()
cnn_dummy.save_weights('headpose_cnn.weights.h5')

# Buat LSTM dummy
lstm_dummy = build_lstm_model()
lstm_dummy.save_weights('focus_lstm.weights.h5')

print("Dummy models created successfully: headpose_cnn.weights.h5 & focus_lstm.weights.h5")
