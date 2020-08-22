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

    criterion = torch.nn.MSELoss(reduction="none")
    _, batch = next(enumerate(test_loader))
    with torch.no_grad():

            outputs = model(batch)

            test_loss = criterion(outputs, batch)

    test_loss = test_loss.mean(dim=1)
    probs = [prob.item() for prob in test_loss]
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
    print(dataset_test[:,:].shape)
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test))

    probs = predict(test_loader)

    return jsonify(probs)


@app.route('/')
def index():
    return "Hello!!:)"


if __name__ == '__main__':
    app.run()
