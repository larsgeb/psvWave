# Use the official image as a parent image.
FROM quay.io/pypa/manylinux1_x86_64

WORKDIR /home

COPY . .

RUN /opt/python/cp37-cp37m/bin/python -m pip install cmake
RUN ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake
RUN /opt/python/cp37-cp37m/bin/python -m pip install -e .[dev]

