FROM nvidia/cuda:10.2-runtime-ubuntu18.04
ENV LANG C.UTF-8

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.8 \
    python3-pip \
    python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install poetry==1.1.12
RUN poetry env use 3.8
RUN poetry install

CMD ["poetry", "run", "python", "./video_trainer/main.py"]
