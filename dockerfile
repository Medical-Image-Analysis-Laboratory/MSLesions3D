
# Contains pytorch, torchvision, cuda, cudnn
FROM nvcr.io/nvidia/pytorch:22.01-py3
 
RUN apt-get update && apt-get install -y git 

WORKDIR /MSLesions3D

RUN mkdir /MSLesions3D/lesions3d

ENV HOME=/MSLesions3D

COPY requirements.txt requirements.txt 

RUN pip3 install -r requirements.txt

RUN git config --global --add safe.directory /code

#COPY --chown=$DOCKER_USER ./lesions3d /MSLesions3D/lesions3d

CMD ["tail", "-f", "/dev/null"]
