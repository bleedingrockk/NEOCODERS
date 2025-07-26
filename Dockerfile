FROM python:3.10-slim

# Install system packages
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-b", ":8080", "app:app"]
