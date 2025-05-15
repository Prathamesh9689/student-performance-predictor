from flask import Flask, render_template
from utils import load_data, preprocess_data
from model import train_model

app = Flask(__name__)

@app.route("/")
def index():
    df = load_data('data/student_data.csv')
    X, y = preprocess_data(df)
    model, acc = train_model(X, y)
    return render_template("index.html", accuracy=round(acc * 100, 2))

if __name__ == "__main__":
    app.run(debug=True)
