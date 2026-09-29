"""Production Gunicorn defaults for the Render web service."""

import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
worker_class = 'uvicorn.workers.UvicornWorker'
workers = 1
accesslog = '-'
