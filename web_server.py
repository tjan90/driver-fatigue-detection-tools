from flask import Flask
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def display():

    return "Looks like it works!"

if __name__=='__main__':
    app.run()