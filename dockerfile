
# Contains pytorch, torchvision, cuda, cudnn
FROM nvcr.io/nvidia/pytorch:22.01-py3
 
RUN apt-get update && apt-get install -y git

WORKDIR /app

RUN pip install pytorch-lightning==1.7.5 && \
	pip install monai==0.9.0 && \
	pip install nibabel && \
	pip install matplotlib && \
	pip install scikit-learn && \
	pip install pandas 

RUN git clone https://github.com/Medical-Image-Analysis-Laboratory/MSLesions3D.git

CMD ["tail", "-f", "/dev/null"]
