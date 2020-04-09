from bokeh.client import pull_session
from bokeh.embed import server_session
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    session = pull_session(url="http://localhost:5006/res_operacionais")
    bokeh_script = server_session(None, url="http://localhost:5006/res_operacionais", session_id=session.id)
    return render_template("dashboard.html", bokeh_script=bokeh_script)


if __name__ == '__main__':
    app.run(debug=True)
