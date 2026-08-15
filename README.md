# llama.cpp TurboQuant + CUDA 13.3 Docker 镜像构建

本项目提供了使用 **GitHub Actions CI/CD** 自动化编译支持 **TurboQuant**（KV Cache 超低比特极限压缩）与 **NVIDIA CUDA 13.3** 的 `llama.cpp` 容器镜像的完整方案。

---

## 🌟 核心特性

1. **TurboQuant KV Cache 压缩**：
   - 基于 Walsh-Hadamard 旋转与极坐标量化算法，将 KV 缓存深度压缩至 **2~4 bits**（如 `turbo4`, `turbo3`, `turbo2`），极大降低长上下文（64K~128K+）显存开销。
   - 支持非对称 K/V（如 `-ctk q8_0 -ctv turbo4`）与边界层保护（Boundary Protection）。
2. **CUDA 13.3 深度加速**：
   - 基于 `nvidia/cuda:13.3.0-devel-ubuntu24.04` 编译，`runtime` 阶段极致轻量化。
   - 支持 FlashAttention (`-DGGML_CUDA_FA_ALL_QUANTS=ON`) 与 CUDA Graph (`-DGGML_CUDA_GRAPHS=ON`)。
   - 适配主流 NVIDIA 架构：Turing (75), Ampere (80/86), Ada Lovelace (89), Hopper (90), Blackwell (100/120)。
3. **GitHub Actions 自动化发布**：
   - 支持手动触发 (`workflow_dispatch`) 灵活指定源码分支、CUDA 版本与架构。
   - 自动推送到 GitHub Container Registry (`ghcr.io`)，并可配置推送到 Docker Hub。
   - 配置 GitHub Actions GHA 构建缓存，后续构建仅需几分钟。

---

## 📁 目录结构

```text
├── .github/
│   └── workflows/
│       └── build-and-push.yml   # GitHub Actions 自动化编译工作流
├── Dockerfile                   # 多阶段编译 Dockerfile (CUDA 13.3 + TurboQuant)
├── docker-compose.yml           # 本地 GPU 容器启动编排文件
├── .dockerignore                # 忽略构建无关文件
└── README.md                    # 项目与运行说明文档
```

---

## 🚀 如何使用 GitHub Actions 编译镜像

### 1. 推送代码到 GitHub
将本仓库推送到您的 GitHub 账号中：
```bash
git init
git add .
git commit -m "feat: add llama.cpp turboquant cuda 13.3 docker build"
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

### 2. 触发编译流程
- 进入 GitHub 仓库页面，点击顶部 **Actions** 标签。
- 在左侧选择 **Build llama.cpp TurboQuant CUDA 13.3 Docker Image**。
- 点击右侧 **Run workflow** 按钮，根据需要调整参数：
  - **NVIDIA CUDA Toolkit Version**: 默认 `13.3.0`
  - **Ubuntu Base Image Version**: 默认 `24.04`
  - **Git Repository URL**: 默认 `https://github.com/TheTom/llama-cpp-turboquant.git`（亦可切换为官方 `ggml-org/llama.cpp.git`）
  - **Git Branch / Tag**: 默认 `master`
  - **Target CUDA Architectures**: 默认 `75;80;86;89;90;100;120`（涵盖 RTX 20/30/40/50 系列、A100、H100、B200）
- 点击 **Run workflow** 开始编译。

### 3. 获取编译产物
构建完成后，镜像将自动发布至 GitHub Packages：
```bash
# 登录 GHCR (如果包为私有)
echo $GITHUB_TOKEN | docker login ghcr.io -u <你的用户名> --password-stdin

# 拉取镜像
docker pull ghcr.io/<你的用户名>/<仓库名>:cuda13.3.0
```

---

## 💻 运行与使用

### 前置要求
- 宿主机已安装 NVIDIA 显卡驱动（CUDA 13.3 推荐驱动版本 `>= 580`）。
- 宿主机已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

### 方式一：使用 Docker CLI 运行 `llama-server`

```bash
docker run -d --name llama-server \
  --gpus all \
  -p 8080:8080 \
  -v /path/to/your/models:/models \
  ghcr.io/<你的用户名>/<仓库名>:cuda13.3.0 \
  -m /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -fa 1 \
  -ctk turbo4 \
  -ctv turbo4 \
  -c 32768
```

### 方式二：使用 Docker Compose 运行

1. 将 GGUF 模型文件放入 `./models` 目录。
2. 修改 `docker-compose.yml` 中的模型文件名与镜像名：
   ```yaml
   image: ghcr.io/<你的用户名>/<仓库名>:cuda13.3.0
   command: >
     -m /models/your-model.gguf
     --host 0.0.0.0
     --port 8080
     -ngl 99
     -fa 1
     -ctk turbo4
     -ctv turbo4
     -c 32768
   ```
3. 启动服务：
   ```bash
   docker compose up -d
   ```
4. 查看运行日志：
   ```bash
   docker compose logs -f
   ```

---

## ⚙️ TurboQuant 启动参数说明

| 参数 | 说明 | 推荐值 |
| :--- | :--- | :--- |
| `-ctk` | Key Cache 量化类型（支持 `turbo4`, `turbo3`, `turbo2`, `q8_0`, `f16` 等） | `turbo4` 或 `q8_0` |
| `-ctv` | Value Cache 量化类型（支持 `turbo4`, `turbo3`, `turbo2`, `q8_0`, `f16` 等） | `turbo4` |
| `-fa 1` | 启用 FlashAttention 加速 | `1` |
| `-ngl 99` | 将所有层卸载至 GPU 显存 | `99` |
| `-c` | 上下文窗口大小（Context Window） | `32768` 或更高 |

> **提示**：若某些特定模型在对称 `-ctk turbo4 -ctv turbo4` 下注意力存在微小偏差，可尝试采用非对称配置：`-ctk q8_0 -ctv turbo4`，在极低显存增加的前提下获得接近 FP16 的精度表现。

---

## 🛠️ 本地直接构建镜像（可选）

如需在本地服务器直接构建镜像，可运行：

```bash
docker build \
  --build-arg CUDA_VERSION=13.3.0 \
  --build-arg UBUNTU_VERSION=24.04 \
  --build-arg CUDA_DOCKER_ARCH="86;89" \
  -t llama-cpp-turboquant:cuda13.3 .
```
