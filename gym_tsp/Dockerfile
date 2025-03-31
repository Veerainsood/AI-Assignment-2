# Use Ubuntu 16.04 for Qt4 support
FROM ubuntu:16.04

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools, dependencies, and Qt4 libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    libboost-all-dev \
    zlib1g-dev \
    flex \
    qt4-dev-tools \
    libqt4-dev \
    && rm -rf /var/lib/apt/lists/*

# Optionally, install additional packages (e.g., for protobuf if needed)
# RUN apt-get update && apt-get install -y libprotobuf-dev protobuf-compiler && rm -rf /var/lib/apt/lists/*

# Clone the HFO repository
WORKDIR /root
RUN git clone https://github.com/LARG/HFO.git

# Build and install HFO (runs ExternalProject_Add targets as defined in CMakeLists.txt)
WORKDIR /root/HFO
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Set working directory and default command
WORKDIR /root/HFO
CMD ["/bin/bash"]
