from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

app = FastAPI()

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("./Data.csv")
df.columns = ["Index", "Height", "Weight"]

# Tạo dữ liệu x và y
x = df["Height"].values
y = df["Weight"].values

# Tổng số mẫu
N = x.shape[0]

# Tính toán hệ số cho Hồi quy tuyến tính
m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) - (np.sum(x) ** 2))
b = (np.sum(y) - m * np.sum(x)) / N

# Hàm dự đoán của hồi quy tuyến tính
def predict_weight(height: float) -> float:
    return m * height + b

# Hàm để tìm alpha tối ưu sử dụng RandomizedSearchCV
def find_best_alpha():
    ridge_model = Ridge()
    param_dist = {'alpha': np.logspace(-3, 3, 100)}
    search = RandomizedSearchCV(ridge_model, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42)
    search.fit(x.reshape(-1, 1), y)
    return search.best_params_['alpha']

# Tìm alpha tốt nhất trước khi huấn luyện mô hình
best_alpha = find_best_alpha()

# Hàm dự đoán cho Ridge Regression
def predict_weight_ridge(height: float) -> float:
    ridge_m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) + best_alpha * N - (np.sum(x) ** 2))
    ridge_b = (np.sum(y) - ridge_m * np.sum(x)) / N
    return ridge_m * height + ridge_b

# Neural Network Regression
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=0)

# Lưu trữ các giá trị mất mát và độ chính xác
losses_linear = []
losses_ridge = []
losses_neural = []
epochs = 100

# Huấn luyện mô hình và lưu giá trị mất mát
def train_models():
    global losses_linear, losses_ridge, losses_neural
    
    # Hồi quy tuyến tính
    for _ in range(epochs):
        pred = [predict_weight(h) for h in x]
        loss = mean_squared_error(y, pred)
        losses_linear.append(loss)

    # Hồi quy Ridge
    for _ in range(epochs):
        pred = [predict_weight_ridge(h) for h in x]
        loss = mean_squared_error(y, pred)
        losses_ridge.append(loss)

    # Huấn luyện Neural Network
    for _ in range(epochs):
        neural_model.fit(x.reshape(-1, 1), y)
        loss = neural_model.loss_
        losses_neural.append(loss)

train_models()

def predict_weight_neural(height: float) -> float:
    return neural_model.predict(np.array([[height]]))[0]

# Định nghĩa lớp dữ liệu đầu vào
class PredictionInput(BaseModel):
    height: float

# Hàm để tạo ma trận nhầm lẫn
def create_confusion_matrix(y_true, y_pred, model_name):
    # Chia thành các khoảng (bins)
    bins = np.linspace(np.min(y), np.max(y), num=5)
    y_true_binned = np.digitize(y_true, bins) - 1
    y_pred_binned = np.digitize(y_pred, bins) - 1
    
    cm = confusion_matrix(y_true_binned, y_pred_binned)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(bins) - 1)
    plt.xticks(tick_marks, [f'{bins[i]:.1f} - {bins[i+1]:.1f}' for i in range(len(bins) - 1)], rotation=45)
    plt.yticks(tick_marks, [f'{bins[i]:.1f} - {bins[i+1]:.1f}' for i in range(len(bins) - 1)])
    
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"./static/confusion_matrix_{model_name}.png")
    plt.close()

def evaluate_model():
    y_pred_linear = [predict_weight(h) for h in x]
    y_pred_ridge = [predict_weight_ridge(h) for h in x]
    y_pred_neural = neural_model.predict(x.reshape(-1, 1))

    metrics = {
        "Linear": {
            "MSE": mean_squared_error(y, y_pred_linear),
            "MAE": mean_absolute_error(y, y_pred_linear),
            "R²": 1 - (np.sum((y - y_pred_linear) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        },
        "Ridge": {
            "MSE": mean_squared_error(y, y_pred_ridge),
            "MAE": mean_absolute_error(y, y_pred_ridge),
            "R²": 1 - (np.sum((y - y_pred_ridge) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        },
        "Neural": {
            "MSE": mean_squared_error(y, y_pred_neural),
            "MAE": mean_absolute_error(y, y_pred_neural),
            "R²": 1 - (np.sum((y - y_pred_neural) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        }
    }

    if not os.path.exists('./static'):
        os.makedirs('./static')

    # Biểu đồ so sánh dự đoán
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Actual", color="black")
    plt.plot(x, y_pred_linear, label="Linear Regression", color="blue")
    plt.plot(x, y_pred_ridge, label="Ridge Regression", color="green")
    plt.plot(x, y_pred_neural, label="Neural Network", color="red")
    plt.legend()
    plt.title("Comparison of Model Predictions")
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.grid(True)
    plt.savefig("./static/predictions_comparison.png")
    plt.close()

    # Biểu đồ mất mát
    plt.figure(figsize=(10, 6))
    plt.plot(losses_linear, label='Linear Regression Loss', color='blue')
    plt.plot(losses_ridge, label='Ridge Regression Loss', color='green')
    plt.plot(losses_neural, label='Neural Network Loss', color='red')
    plt.legend()
    plt.title("Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')  # Sử dụng log scale để dễ dàng theo dõi hội tụ
    plt.grid(True)
    plt.savefig("./static/loss_plot.png")
    plt.close()

    # Tạo ma trận nhầm lẫn
    create_confusion_matrix(y, y_pred_linear, "Linear")
    create_confusion_matrix(y, y_pred_ridge, "Ridge")
    create_confusion_matrix(y, y_pred_neural, "Neural")

    return metrics

@app.post("/predict")
async def predict(input_data: PredictionInput):
    height = input_data.height

    predicted_weight_linear = predict_weight(height)
    predicted_weight_ridge = predict_weight_ridge(height)
    predicted_weight_neural = predict_weight_neural(height)

    metrics = evaluate_model()

    return {
        "predicted_weight_linear": predicted_weight_linear,
        "predicted_weight_ridge": predicted_weight_ridge,
        "predicted_weight_neural": predicted_weight_neural,
        "metrics": metrics,
        "chart_url": "/static/predictions_comparison.png",
        "loss_chart_url": "/static/loss_plot.png",
        "confusion_matrix_linear_url": "/static/confusion_matrix_Linear.png",
        "confusion_matrix_ridge_url": "/static/confusion_matrix_Ridge.png",
        "confusion_matrix_neural_url": "/static/confusion_matrix_Neural.png",
    }

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <html>
        <head>
            <title>Dự Đoán Cân Nặng</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container">
                <div class="row justify-content-center align-items-center" style="height: 100vh;">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h2 class="card-title text-center">Dự đoán cân nặng</h2>
                                <form id="predictionForm">
                                    <div class="form-group">
                                        <label for="height">Chiều cao (cm):</label>
                                        <input type="number" id="height" name="height" class="form-control" step="any" required>
                                    </div>
                                    <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
                                </form>
                                <div id="result" class="mt-3"></div>
                                <img id="chart" src="" class="mt-3" style="width:100%;">
                                <img id="loss_chart" src="" class="mt-3" style="width:100%;">
                                <img id="confusion_matrix_linear" src="" class="mt-3" style="width:100%;">
                                <img id="confusion_matrix_ridge" src="" class="mt-3" style="width:100%;">
                                <img id="confusion_matrix_neural" src="" class="mt-3" style="width:100%;">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <script>
                async function predict() {
                    const height = document.getElementById('height').value;

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ height: parseFloat(height) }),
                    });

                    const data = await response.json();
                    document.getElementById('result').innerHTML = `
                        <p>Cân nặng dự đoán theo Hồi quy tuyến tính: ${data.predicted_weight_linear.toFixed(2)} kg</p>
                        <p>Cân nặng dự đoán theo Hồi quy Ridge: ${data.predicted_weight_ridge.toFixed(2)} kg</p>
                        <p>Cân nặng dự đoán theo Neural Network: ${data.predicted_weight_neural.toFixed(2)} kg</p>
                        <p><strong>Đánh giá mô hình:</strong></p>
                        <p>Hồi quy tuyến tính - MSE: ${data.metrics.Linear.MSE.toFixed(2)}, MAE: ${data.metrics.Linear.MAE.toFixed(2)}, R²: ${data.metrics.Linear["R²"].toFixed(2)}</p>
                        <p>Hồi quy Ridge - MSE: ${data.metrics.Ridge.MSE.toFixed(2)}, MAE: ${data.metrics.Ridge.MAE.toFixed(2)}, R²: ${data.metrics.Ridge["R²"].toFixed(2)}</p>
                        <p>Neural Network - MSE: ${data.metrics.Neural.MSE.toFixed(2)}, MAE: ${data.metrics.Neural.MAE.toFixed(2)}, R²: ${data.metrics.Neural["R²"].toFixed(2)}</p>
                    `;

                    document.getElementById('chart').src = data.chart_url;
                    document.getElementById('loss_chart').src = data.loss_chart_url;
                    document.getElementById('confusion_matrix_linear').src = data.confusion_matrix_linear_url;
                    document.getElementById('confusion_matrix_ridge').src = data.confusion_matrix_ridge_url;
                    document.getElementById('confusion_matrix_neural').src = data.confusion_matrix_neural_url;
                }
            </script>
        </body>
    </html>
    """

@app.get("/static/{file_name}")
async def static_files(file_name: str):
    return FileResponse(f"./static/{file_name}")
