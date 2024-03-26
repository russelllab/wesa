from flask import Flask
from celery import Celery

this_app = Flask(__name__)
this_app.config['broker_url'] = 'redis://localhost:6379/0'
this_app.config['result_backend'] = 'redis://localhost:6379/0'
this_app.config['broker_connection_retry_on_startup'] = True

from wesa_app import main
