# Bokeh Dashboard

This repo contains an example of the implementation of a Python dashboard using the Bokeh library. It is deployed using a simple Flask app. For styling, Bootstrap was used as a convenient and practical way to add attractive layout to Flask apps.



## Running the Dashboard

There are only a few steps required to run this implementation of a Bokeh dashboard on your local machine. The steps are the following:


2. Make sure you have a virtual environment with <em>flask</em> and <em>bokeh</em> installed.
3. Open two command terminal.
4. In the first window run the command 
bokeh serve --allow-websocket-origin=127.0.0.1:5000 res_operacionais.py
5. In the second window run the command "python app.py"
6. Finally, open the localhost - usually at http://127.0.0.1:5000/
