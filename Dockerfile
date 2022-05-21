FROM python:3.9
WORKDIR /app
RUN pip install --upgrade pip setuptools 
RUN pip  install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip uninstall opencv-python-headless -y
RUN pip install opencv-python==4.4.0.46

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]   