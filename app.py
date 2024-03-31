import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default port or one provided by Cloud Run environment
    app.run(debug=True, host='0.0.0.0', port=port)
