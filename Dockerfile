FROM nvcr.io/nvidia/pytorch:20.11-py3

# Step1 build yolor & AIDMS env

RUN apt-get update && apt-get install -y sudo openssh-server \
    && service ssh start

RUN groupadd -g 1000 ubuntu \
    && useradd -rm -d /home/ubuntu -s /bin/bash -g 1000 -u 1000 ubuntu  -p "$(openssl passwd -1 ubuntu)" \
    && usermod -aG sudo ubuntu 


RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg git vim curl \
    && apt-get install -y zip htop screen libgl1-mesa-glx \
    && apt-get clean 

RUN echo "export PATH=$PATH:/opt/conda/bin" >> /root/.bashrc
RUN echo "export PATH=$PATH:/opt/conda/bin" >> /home/ubuntu/.bashrc

RUN rm /usr/bin/gcc \
    && apt-get install gcc-4.8 -y && ln -s /usr/bin/gcc-4.8 /usr/bin/gcc && pip3 install uwsgi \
    && ln -s /opt/conda/bin/tensorboard /usr/local/bin/tensorboard \
    && chown -R ubuntu:ubuntu /workspace 

USER ubuntu
WORKDIR /workspace

#AIDMS && YOLOR
RUN pip3 install labelme2coco declxml torchsummary pyyaml && pip3 uninstall opencv-pythonsu u && pip3 install opencv-python-headless \
    && pip3 install seaborn 

# # step2 copy source code & pretrain weight
# aidms-models : git clone http://192.168.1.228:3000/LEADTEK/aidms-models.git -b model/yolor
COPY --chown=ubuntu:ubuntu aidms ./aidms
COPY --chown=ubuntu:ubuntu customized ./customized

# # pretrain weight
RUN bash customized/models/yolor/scripts/get_pretrain.sh \
    && mv yolor_w6.pt customized/models/pre_weight \
    && mv yolor_p6.pt customized/models/pre_weight 
