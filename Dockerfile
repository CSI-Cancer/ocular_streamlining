# syntax=docker/dockerfile:1

# Build with (while in Dockerfile folder):
# $ docker build -t ocular_streamlining .
# Start container to run stuff detached, with a volume mounted, and then delete itself:
# $ docker run -d --rm -v [HOST_PATH]:[CONTAINER_PATH] ocular_streamlining command
# Interactive example:
# $ docker run -it --rm -v /mnt/HDSCA_Development:/mnt/HDSCA_Development -v ./models:/ocular_streamlining/models ocular_streamlining


ARG PYTHON_VERSION=3.11-slim-bookworm

# https://hub.docker.com/_/python
FROM python:$PYTHON_VERSION

# ARGs are erased after FROM statements, so these need to be here
ARG PACKAGE_NAME=ocular_streamlining

WORKDIR /$PACKAGE_NAME

# To avoid odd requests during apt install
ENV DEBIAN_FRONTEND=noninteractive

## Prepare venv
#RUN python -m venv /venv
#ENV PATH=/venv/bin:$PATH

# Copy over private dependencies
COPY --from=csi_utils /csi_utils /csi_utils
COPY --from=csi_analysis /csi_analysis /csi_analysis

# Copy over package and install
COPY $PACKAGE_NAME /$PACKAGE_NAME/$PACKAGE_NAME
COPY streamlining_training /$PACKAGE_NAME/streamlining_training
COPY scripts /$PACKAGE_NAME/scripts
#COPY examples /$PACKAGE_NAME/examples
#COPY tests /$PACKAGE_NAME/tests
COPY pyproject.toml requirements.txt /$PACKAGE_NAME/
# Poetry does not require a venv (and it's harder to use one here...)
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install

ENTRYPOINT ["bash"]
