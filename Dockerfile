FROM python:3.9-slim-bullseye AS base
FROM base AS builder

RUN useradd mariotte
RUN mkdir  /install
WORKDIR /install
COPY requirements.txt requirements.txt
RUN pip install --prefix=/install -r requirements.txt

FROM base
COPY --from=builder /install /usr/local
ENV APP=/home/mariotte/app
WORKDIR $APP
COPY . .
CMD ["python3", "main.py"]
