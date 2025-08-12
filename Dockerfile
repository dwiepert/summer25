# use a Google maintained base image hosted in 
# Google's container registry (https://console.cloud.google.com/gcr/images/deeplearning-platform-release)
FROM gcr.io/deeplearning-platform-release/pytorch-cu124-conda.2-4.py310

# pip requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#Clone and build summer25
RUN git clone https://github.com/dwiepert/summer25.git && \
    cd summer25 && \
    pip install .

#execute code
ENTRYPOINT ["python", "summer25/run.py"]