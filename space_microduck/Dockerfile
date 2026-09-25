# A Docker Space rather than the Gradio SDK: the SDK installs gradio[mcp], whose mcp<2 pin cannot coexist with
# quackd 0.14.0's mcp>=2, so gradio goes in here without that extra. Everything else is what the SDK would do.
FROM python:3.12-slim

# OSMesa: MuJoCo renders the 3D simulator headless through it. git: pip installs laya from a pinned commit.
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt gradio==6.28.0
COPY --chown=user app.py .

EXPOSE 7860
CMD ["python", "app.py"]
