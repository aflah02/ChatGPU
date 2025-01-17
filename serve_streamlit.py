# Based on - https://modal.com/docs/examples/serve_streamlit

import shlex
import subprocess
from pathlib import Path

import modal

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = "/root/app.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("streamlit==1.41.0", "openai")
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
)

app = modal.App(name="ChatGPU", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

@app.function(
    allow_concurrent_inputs=100,
)
@modal.web_server(8000)
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)