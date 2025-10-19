FROM python:3.12-slim
USER root

RUN apt-get update && apt-get install -y libmagic1 curl && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.cargo/bin:$PATH"


RUN mkdir -p /knowledge-graphs
WORKDIR /knowledge-graphs

COPY requirements.txt ./requirements.txt
RUN uv pip install --system -r requirements.txt

COPY src ./src
COPY app.py ./app.py
COPY config_example.env ./config_example.env

EXPOSE 8000

ENTRYPOINT ["uv", "run", "fastapi", "run", "backend.py", "--server.port=8000", "server.address=0.0.0.0"]