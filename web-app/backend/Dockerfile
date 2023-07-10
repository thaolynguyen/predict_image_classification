FROM python:3.9-slim 

# RUN apt-get update && apt-get install -y python3 python3-distutils python3-pip python3-apt

WORKDIR /app

# RUN echo \
#    && apt-get update \
#    && apt-get --yes install apt-file \
#    && apt-file update
# RUN echo \
#    && apt-get --yes install build-essential
# ARG USER=nobody
# RUN usermod -aG sudo $USER


COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install tensorflow && \
    pip install keras_applications==1.0.4 --no-deps && \
    pip install keras_preprocessing==1.0.2 --no-deps 
    # pip install h5py==2.8.0

RUN pip3 install --upgrade protobuf==3.20.0

COPY . .

EXPOSE 5001

CMD [ "python", "server.py", "--reload"]