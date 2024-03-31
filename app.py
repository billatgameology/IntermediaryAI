from flask import Flask
import os

app = Flask(__name__)

port = int(os.environ.get('PORT', 8080))
app.run(host='0.0.0.0', port=port)

@app.route('/')
def hello_world():
    return 'Hello, World!'

