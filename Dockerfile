# ==============================================================================
# Multi-stage Dockerfile for llama.cpp with TurboQuant and CUDA 13.3+ Support
# ==============================================================================

ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=24.04

# ------------------------------------------------------------------------------
# Stage 1: Build Environment
# ------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# Build arguments
ARG REPO_URL=https://github.com/TheTom/llama-cpp-turboquant.git
ARG REPO_REF=master
ARG CUDA_DOCKER_ARCH="75;80;86;89;90;100;120"
ARG CMAKE_EXTRA_FLAGS="-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_GRAPHS=ON"

# Set environment
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}

# Install essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    ca-certificates \
    curl \
    pkg-config \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the target llama.cpp repository (TurboQuant fork or official)
WORKDIR /src
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} llama.cpp \
    || (git clone ${REPO_URL} llama.cpp && cd llama.cpp && git checkout ${REPO_REF})

WORKDIR /src/llama.cpp

# Configure and compile with CMake
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_BUILD_TESTS=OFF \
    -DGGML_BUILD_EXAMPLES=ON \
    -DGGML_BUILD_SERVER=ON \
    -DGGML_CURL=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
    ${CMAKE_EXTRA_FLAGS} \
    && cmake --build build --config Release -j$(nproc)

# Stage all built binaries into a single directory for easy transfer
RUN mkdir -p /dist/bin /dist/lib && \
    find build/bin -maxdepth 1 -type f -executable -exec cp {} /dist/bin/ \; || true && \
    find build -name "*.so*" -exec cp -P {} /dist/lib/ \; 2>/dev/null || true

# ------------------------------------------------------------------------------
# Stage 2: Runtime Environment
# ------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runner

LABEL maintainer="Antigravity AI" \
      description="llama.cpp with TurboQuant KV Cache Compression & CUDA 13.3+ Acceleration" \
      org.opencontainers.image.source="https://github.com/TheTom/llama-cpp-turboquant"

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    libstdc++6 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries and libraries from builder
COPY --from=builder /dist/bin/ /usr/local/bin/
COPY --from=builder /dist/lib/ /usr/local/lib/
RUN ldconfig

# Create non-root user and directory for model weights
RUN useradd -m -u 1000 llama && \
    mkdir -p /models && \
    chown -R llama:llama /models

WORKDIR /models
EXPOSE 8080

# Healthcheck for llama-server
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${LLAMA_ARG_PORT}/health || exit 1

# Default to running llama-server
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
