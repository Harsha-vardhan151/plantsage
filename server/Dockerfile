FROM python:3.10-slim
WORKDIR /app
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /app
ENV TORCH_DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]