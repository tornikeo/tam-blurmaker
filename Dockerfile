FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN apt update
# RUN apt update && apt install ffmpeg git -y
RUN apt-get install git -y
RUN apt-get install ffmpeg -y
WORKDIR /workdir
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY download.py .
RUN python download.py
COPY . .
# ENTRYPOINT python cli.py
CMD python cli.py
