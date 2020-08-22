from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.autoencoder import Autoencoder
from data.traffic_data_generator import TrafficDataset

DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

checkpoint = torch.load("./model/model_latest.pth")
model = Autoencoder()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def predict(test_loader: DataLoader) -> dict:
    """
    Evaluate model on the received dataset.

    Args:
        test_loader: received dataset

    Returns:
        dict with the probabilities for each row in the test_loader
    """
    probs = dict()

    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            outputs = model(batch)

            test_loss = criterion(outputs, batch)

            probs[str(i)] = str(test_loss.item())
    return probs


# Sanity check route
@app.route('/ping', methods=['GET'])
@cross_origin()
def ping():
    return jsonify('pong!')


@app.route('/predict', methods=['POST'])
@cross_origin()
def pong():
    file = request.files['file']
    data = pd.read_csv(file)
    dataset_test = TrafficDataset(data)
    test_loader = DataLoader(dataset_test, batch_size=1)

    probs = predict(test_loader)

    return jsonify(probs)


@app.route('/')
def index():
    return "Hello!!:)"


if __name__ == '__main__':
    app.run()
