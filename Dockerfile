FROM python:3.9-slim

# Install Rust
RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . /app

CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8080"]
