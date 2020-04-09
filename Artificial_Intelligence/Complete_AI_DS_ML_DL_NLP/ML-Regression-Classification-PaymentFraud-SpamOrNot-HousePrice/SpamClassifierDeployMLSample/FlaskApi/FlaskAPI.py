from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def first():
    return "Hello, World!"


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/rakesh")
def rakesh():
    return "Rakesh is a Data Scientist Developer"


if __name__ == "__main__":
    app.run(debug=True)
