from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Sanity check route
@app.route('/ping', methods=['GET'])
@cross_origin()
def ping():
    return jsonify('pong!')


@app.route('/pong', methods=['POST'])
@cross_origin()
def pong():
    file = request.files['file']
    read = file.read()
    return read


@app.route('/')
def index():
    return "Hello:)"


if __name__ == '__main__':
    app.run()
