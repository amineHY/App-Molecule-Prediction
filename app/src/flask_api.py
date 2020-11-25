from flask import Flask
from app.src.main import Predict

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/predict/<path:path_X_test>')
def predict(path_X_test):
    """
    Example: 
    path_X_test=app/data/dataset_single.csv
    Link: 
        http://127.0.0.1:5000/predict/app/data/dataset_single.csv
    """
    isinstance(path_X_test, str)
    y_pred = Predict(path_X_test)
    return {'status': 'OK', 'y_pred': y_pred.tolist()}


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
