# app.py
import os
import io
import csv
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file, url_for

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import plotly.graph_objs as go
import plotly.io as pio

from model_utils import add_indicators, create_sequences

app = Flask(__name__)
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ------------------------------
# Model builders
# ------------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.15),
        LSTM(32),
        Dropout(0.15),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.15),
        GRU(32),
        Dropout(0.15),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ------------------------------
# Data prep
# ------------------------------
def prepare_data(ticker, period="5y", seq_len=60):
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        return None
    df = add_indicators(df)
    df = df.dropna()
    values = df.values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = create_sequences(scaled, seq_len)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test, scaler, df

# ------------------------------
# Load or train model
# ------------------------------
def load_or_train(model_name, builder_fn, input_shape, X_train, y_train, epochs=7):
    path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    if os.path.exists(path):
        model = builder_fn(input_shape)
        model.load_weights(path)
        return model
    model = builder_fn(input_shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32,
              validation_split=0.1, callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
              verbose=1)
    model.save(path)
    return model

# ------------------------------
# Predict future
# ------------------------------
def predict_future(model, last_scaled, n_days, scenario_factor=0.0):
    seq = last_scaled.copy()
    preds = []
    for _ in range(n_days):
        inp = seq[-60:].reshape(1, 60, seq.shape[1])
        p = model.predict(inp, verbose=0)[0][0]
        p = p * (1 + scenario_factor)
        new_row = np.zeros(seq.shape[1])
        new_row[0] = p
        seq = np.vstack([seq, new_row])
        preds.append(p)
    return preds

# ------------------------------
# Compute metrics
# ------------------------------
def compute_metrics(y_true, y_pred, scaler=None, df_sample=None):
    if scaler is not None and df_sample is not None:
        dummy_true = np.zeros((len(y_true), df_sample.shape[1]))
        dummy_pred = np.zeros_like(dummy_true)
        dummy_true[:,0] = y_true.flatten()
        dummy_pred[:,0] = y_pred.flatten()
        y_true_real = scaler.inverse_transform(dummy_true)[:,0]
        y_pred_real = scaler.inverse_transform(dummy_pred)[:,0]
    else:
        y_true_real = y_true.flatten()
        y_pred_real = y_pred.flatten()
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mape = np.mean(np.abs((y_true_real - y_pred_real) / (y_true_real + 1e-9))) * 100
    mean_price = np.mean(y_true_real) if len(y_true_real)>0 else 1.0
    confidence = max(0.0, 1 - (rmse / (mean_price + 1e-9)))
    return {"mae": mae, "rmse": rmse, "mape": mape, "confidence": confidence}

# ------------------------------
# Plot helpers (Plotly)
# ------------------------------
def plot_history_plotly(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=400, template="plotly_white",
                      title="Historical Close Price")
    return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

def plot_test_vs_pred(y_test_real, lstm_real, gru_real):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test_real, mode='lines', name='Original Test'))
    fig.add_trace(go.Scatter(y=lstm_real, mode='lines', name='LSTM Pred', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(y=gru_real, mode='lines', name='GRU Pred', line=dict(dash='dot')))
    fig.update_layout(height=420, template="plotly_white", title="Original vs Predicted (Test Set)", margin=dict(l=20,r=20,t=30,b=20))
    return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

def plot_future_plotly(dates, future_vals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=future_vals, mode='lines+markers', name='Future Pred'))
    fig.update_layout(height=380, template="plotly_white", title="Future Predictions", margin=dict(l=20,r=20,t=30,b=20))
    return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

# ------------------------------
# Download CSV
# ------------------------------
@app.route("/download", methods=["GET"])
def download():
    dates = request.args.getlist("date")
    prices = request.args.getlist("price")
    if not dates or not prices or len(dates)!=len(prices):
        return "Missing data", 400
    mem = io.StringIO()
    writer = csv.writer(mem)
    writer.writerow(["date","predicted_price"])
    for d,p in zip(dates, prices):
        writer.writerow([d,p])
    mem.seek(0)
    return send_file(io.BytesIO(mem.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ------------------------------
# Main route
# ------------------------------
@app.route("/", methods=["GET","POST"])
def index():
    tickers = ["BTC-USD","ETH-USD","SOL-USD"]
    if request.method == "POST":
        ticker = request.form.get("stock").strip() or "BTC-USD"
        if ticker not in tickers:
            ticker = "BTC-USD"
        try:
            n_days = int(request.form.get("no_of_days", 10))
        except:
            n_days = 10
        scenario = request.form.get("scenario", "neutral")
        model_choice = request.form.get("model_choice", "best")
        scenario_map = {"bullish":0.02, "bearish":-0.03, "neutral":0.0}
        factor = scenario_map.get(scenario, 0.0)

        # Prepare data
        data = prepare_data(ticker, period="5y", seq_len=60)
        if data is None:
            return render_template("index.html", error="Invalid ticker or no data found")
        X_train, y_train, X_test, y_test, scaler, df = data
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Load models
        lstm = load_or_train("lstm", build_lstm, input_shape, X_train, y_train)
        gru = load_or_train("gru", build_gru, input_shape, X_train, y_train)

        # Test predictions
        lstm_pred = lstm.predict(X_test)
        gru_pred = gru.predict(X_test)
        baseline_pred = X_test[:, -1, 0].reshape(-1,1)

        metrics_lstm = compute_metrics(y_test, lstm_pred, scaler=scaler, df_sample=df)
        metrics_gru = compute_metrics(y_test, gru_pred, scaler=scaler, df_sample=df)
        metrics_base = compute_metrics(y_test, baseline_pred, scaler=scaler, df_sample=df)

        # Choose model
        if model_choice=="lstm": chosen_model=lstm
        elif model_choice=="gru": chosen_model=gru
        elif model_choice=="baseline": chosen_model=None
        else: chosen_model=lstm if metrics_lstm['mae']<metrics_gru['mae'] else gru

        # Future predictions
        last_scaled = scaler.transform(df.values)[-60:]
        if chosen_model is None:
            last_close_scaled = last_scaled[-1,0]
            future_scaled = [last_close_scaled for _ in range(n_days)]
        else:
            future_scaled = predict_future(chosen_model, last_scaled, n_days, factor)
        dummy = np.zeros((len(future_scaled), df.shape[1]))
        dummy[:,0] = future_scaled
        future_real = scaler.inverse_transform(dummy)[:,0]
        future_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1,n_days+1)]
        future_data = list(zip(future_dates, [float(x) for x in future_real]))

        dummy_true = np.zeros((len(y_test), df.shape[1])); dummy_pred_l = np.zeros_like(dummy_true); dummy_pred_g = np.zeros_like(dummy_true)
        dummy_true[:,0] = y_test.flatten(); dummy_pred_l[:,0]=lstm_pred.flatten(); dummy_pred_g[:,0]=gru_pred.flatten()
        y_test_real = scaler.inverse_transform(dummy_true)[:,0]; lstm_real = scaler.inverse_transform(dummy_pred_l)[:,0]; gru_real = scaler.inverse_transform(dummy_pred_g)[:,0]

        hist_plot = plot_history_plotly(df)
        test_plot = plot_test_vs_pred(y_test_real, lstm_real, gru_real)
        future_plot = plot_future_plotly(future_dates, future_real)

        summary = {
            "best_model":"LSTM" if metrics_lstm['mae']<metrics_gru['mae'] else "GRU",
            "lstm_mae":metrics_lstm['mae'],
            "gru_mae":metrics_gru['mae'],
            "baseline_mae":metrics_base['mae'],
            "best_confidence":round(metrics_lstm['confidence']*100 if metrics_lstm['mae']<metrics_gru['mae'] else metrics_gru['confidence']*100,2)
        }

        return render_template("result.html",
                               ticker=ticker,
                               hist_plot=hist_plot,
                               test_plot=test_plot,
                               future_plot=future_plot,
                               future_data=future_data,
                               summary=summary,
                               tickers=tickers)
    return render_template("index.html", tickers=tickers)

if __name__ == "__main__":
    app.run(debug=True)
