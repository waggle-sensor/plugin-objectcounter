FROM waggle/plugin-base:1.1.1-ml-torch1.9.0

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/
COPY coco.names /app/
COPY models/ /app/models
COPY utils/ /app/utils

ADD https://web.lcrc.anl.gov/public/waggle/models/vehicletracking/yolov7.pt /app/yolov7.pt

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
