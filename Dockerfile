# ==============================================================================
# Multi-stage Dockerfile for llama.cpp with TurboQuant and CUDA 13.3+
# Tailored and Optimized for NVIDIA GeForce RTX 50-Series (Blackwell, sm_120)
# ==============================================================================

ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=24.04

# ------------------------------------------------------------------------------
# Stage 1: Build Environment
# ------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# Build arguments - Targeted exclusively for RTX 50-series (Compute Capability 12.0)
ARG REPO_URL=https://github.com/TheTom/llama-cpp-turboquant.git
ARG REPO_REF=master
ARG CUDA_DOCKER_ARCH="120"
ARG CMAKE_EXTRA_FLAGS="-DGGML_CUDA_FA_ALL_QUANTS=ON"

# Set environment
ENV DEBIAN_FRONTEND=noninteractive \
    CCACHE_DIR=/root/.ccache \
    PATH="/usr/lib/ccache:$PATH"

# Install essential build dependencies + ccache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    ca-certificates \
    curl \
    pkg-config \
    libcurl4-openssl-dev \
    libgomp1 \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Clone target repository
WORKDIR /src
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} llama.cpp \
    || (git clone ${REPO_URL} llama.cpp && cd llama.cpp && git checkout ${REPO_REF})

WORKDIR /src/llama.cpp

# Configure and compile with CMake + ccache for RTX 50-series (sm_120)
RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DGGML_CURL=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
        ${CMAKE_EXTRA_FLAGS} \
        . && \
    cmake --build build --config Release -j$(nproc)

# Stage built binaries and shared libraries into /dist
RUN mkdir -p /dist/bin /dist/lib && \
    find build/bin -maxdepth 1 -type f -executable -exec cp {} /dist/bin/ \; || true && \
    find build -name "*.so*" -exec cp -P {} /dist/lib/ \; 2>/dev/null || true

# ------------------------------------------------------------------------------
# Stage 2: Runtime Environment
# ------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runner

LABEL maintainer="Antigravity AI" \
      description="llama.cpp with TurboQuant KV Cache Compression & CUDA 13.3 (RTX 50-Series Blackwell Optimized)" \
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

ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
