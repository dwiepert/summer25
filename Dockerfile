# use a Google maintained base image hosted in 
# Google's container registry (https://console.cloud.google.com/gcr/images/deeplearning-platform-release)
FROM gcr.io/deeplearning-platform-release/base-gpu.py310

COPY /run_jupyter.sh /run_jupyter.sh
RUN chmod +x /run_jupyter.sh

# # pip requirements
RUN pip install --upgrade pip
#RUN pip uninstall torchaudio
#RUN conda uninstall torchaudio

#RUN conda install torchvision==0.19.0 #torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# #Clone and build summer25
RUN git clone https://github.com/dwiepert/summer25.git && \
    cd summer25 && \
    pip install .

# #execute code
#ENTRYPOINT ["python", "summer25/run.py"]