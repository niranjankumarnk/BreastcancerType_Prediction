from flask import Flask , render_template, Request
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
import os

app = Flask(__name__)  #Initializing flask app


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/train', methods=['GET'])
def training():
    os.system('python main.py')
    return "Training Successful !"


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)