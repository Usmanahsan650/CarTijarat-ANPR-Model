FROM python:3.9-slim-bullseye
WORKDIR /app
RUN pip  install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 5000
# RUN apt-get update
# RUN apt install -y libgl1-mesa-glx
RUN pip uninstall opencv-python-headless -y
RUN pip install opencv-python-headless==4.4.0.46
CMD gunicorn main:app --timeout 1000 --workers 1