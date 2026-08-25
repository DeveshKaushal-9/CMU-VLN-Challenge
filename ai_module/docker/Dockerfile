FROM zhangjicmu/ubuntu24_ros:ai_module

USER root

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: a 183MB wheel against multiple GB for the CUDA build. The
# simulation evaluation machine has no discrete GPU, so CPU is the target.
RUN pip3 install --no-cache-dir --break-system-packages \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision

RUN pip3 install --no-cache-dir --break-system-packages transformers pillow

# Bake the detector weights into the image so nothing is fetched at run time.
ENV HF_HOME=/opt/hf
RUN python3 -c "\
from transformers import Owlv2Processor, Owlv2ForObjectDetection; \
Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble'); \
Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')" \
    && chmod -R a+rX /opt/hf

# Never reach for the network during evaluation.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

USER docker

# Replace the stock dummy_vlm with this team's AI module.
RUN rm -rf /home/docker/ai_module/src/dummy_vlm
COPY --chown=docker:docker src/dummy_vlm /home/docker/ai_module/src/dummy_vlm

WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm"

RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
