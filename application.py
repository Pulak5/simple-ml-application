from flask import Flask, request
import pickle
import numpy as np

application = Flask(__name__)



@application.route("/")
def home():
    return "This is home page"


if __name__ == "__main__":
    application.run(debug=True)