FROM python:3.9-slim-bullseye
RUN useradd mariotte
ENV APP=/home/mariotte/app
WORKDIR $APP
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "main.py"]
