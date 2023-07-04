FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    git \
    tzdata \
    libgl1-mesa-glx
ENV TZ Asia/Seoul
RUN git clone https://github.com/LHyunn/rb_streamlit .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 8888
HEALTHCHECK CMD curl --fail http://localhost:8888/_stcore/health
ENTRYPOINT ["streamlit", "run", "RB_Review.py", "--server.port=8888", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]