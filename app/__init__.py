from flask import Flask
import os

app = Flask(__name__)
# Disable the debugger PIN
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

from app import routes

