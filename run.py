import os
import uvicorn

if __name__ == '__main__':
    # Render supplies PORT and requires the process to listen on all interfaces.
    # Keep reload opt-in so production deploys do not spawn a file-watcher process.
    port = int(os.getenv('PORT', '8000'))
    reload_enabled = os.getenv('UVICORN_RELOAD', '').lower() in {'1', 'true', 'yes'}
    uvicorn.run('backend.main:app', host='0.0.0.0', port=port, reload=reload_enabled)
