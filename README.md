# llama.cpp TurboQuant + CUDA Docker 镜像构建

本项目提供了使用 **GitHub Actions CI/CD** 自动化编译支持 **TurboQuant**（KV Cache 超低比特极限压缩）与 **NVIDIA CUDA** 的 `llama.cpp` 容器镜像的完整方案。

---

## 🌟 核心特性

1. **TurboQuant KV Cache 压缩**：
   - 基于 Walsh-Hadamard 旋转与极坐标量化算法，将 KV 缓存深度压缩至 **2~4 bits**（如 `turbo4`, `turbo3`, `turbo2`），极大降低长上下文（64K~128K+）显存开销。
   - 支持非对称 K/V（如 `-ctk q8_0 -ctv turbo4`）与边界层保护（Boundary Protection）。
2. **CUDA 深度加速**：
   - 支持 `13.3.0` / `12.8.1` 等多种 CUDA 版本，`runtime` 阶段极致轻量化。
   - 支持 FlashAttention 全量化加速 (`-DGGML_CUDA_FA_ALL_QUANTS=ON`) 与 CUDA Graph。
   - 预编译适配主流 GPU 架构：Turing (75), Ampere (80/86), Ada Lovelace (89), Hopper (90)。
3. **GitHub Actions 自动化发布**：
   - 支持手动触发 (`workflow_dispatch`) 灵活指定源码分支、CUDA 版本与架构。
   - 自动推送到 GitHub Container Registry (`ghcr.io`)，并可配置推送到 Docker Hub。
   - 配置 GitHub Actions GHA 构建缓存，大幅缩短后续构建时间。

---

## 📁 目录结构

```text
├── .github/
│   └── workflows/
│       └── build-and-push.yml   # GitHub Actions 自动化编译工作流
├── Dockerfile                   # 多阶段编译 Dockerfile (CUDA + TurboQuant)
├── docker-compose.yml           # 本地 GPU 容器启动编排文件
├── .dockerignore                # 忽略构建无关文件
└── README.md                    # 项目与运行说明文档
```

---

## 🚀 如何使用 GitHub Actions 编译镜像

### 1. 开启 GitHub Actions 写入权限（关键！）
在推送代码前，请确保仓库允许 Actions 推送容器镜像到 GitHub Packages (GHCR)：
1. 打开您的 GitHub 仓库页面。
2. 点击 **Settings** -> 左侧 **Actions** -> **General**。
3. 滚动到 **Workflow permissions** 部分，勾选 **Read and write permissions**。
4. 点击 **Save** 保存。

### 2. 推送代码到 GitHub
```bash
git init
git add .
git commit -m "feat: add llama.cpp turboquant cuda docker build"
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

### 3. 触发编译流程
- 进入 GitHub 仓库页面，点击顶部 **Actions** 标签。
- 在左侧选择 **Build llama.cpp TurboQuant CUDA Docker Image**。
- 点击右侧 **Run workflow** 按钮，根据需要调整参数：
  - **NVIDIA CUDA Toolkit Version**: 默认 `13.3.0`（如需兼容旧驱动亦可填 `12.8.1`）
  - **Ubuntu Base Image Version**: 默认 `24.04`
  - **Git Repository URL**: 默认 `https://github.com/TheTom/llama-cpp-turboquant.git`
  - **Git Branch / Tag**: 默认 `master`
  - **Target CUDA Architectures**: 默认 `75;80;86;89;90`
- 点击 **Run workflow** 开始编译。

### 4. 获取编译产物
构建完成后，镜像将自动发布至 GitHub Packages：
```bash
# 登录 GHCR (如果包为私有)
echo $GITHUB_TOKEN | docker login ghcr.io -u <你的用户名> --password-stdin

# 拉取镜像
docker pull ghcr.io/<你的用户名>/<仓库名>:cuda13.3.0
```

---

## 🔍 常见构建失败排查指南

| 常见错误现象 | 原因分析 | 解决办法 |
| :--- | :--- | :--- |
| **`403 Forbidden` / `permission_denied`** | GITHUB_TOKEN 没有包写入权限 | 前往仓库 **Settings** -> **Actions** -> **General** -> **Workflow permissions**，选择 **Read and write permissions** 并保存。 |
| **`nvcc fatal: Value '120' is not defined`** | 指定了当前 nvcc 编译器未支持的过高架构代号 | 将 `CUDA_ARCH` 参数改为 `75;80;86;89;90` 或 `default`。 |
| **`no space left on device`** | GitHub Runner 默认 14GB 磁盘空间不足 | 工作流已加入 `Free disk space on runner` 步骤清理无用预装组件。 |
| **`manifest unknown` / 基础镜像拉取失败** | Docker Hub 标签变更 | 可在触发工作流时将 CUDA 版本调整为具体存在的补丁版本（如 `13.3.1` 或 `12.8.1`）。 |

---

## 💻 运行与使用

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
2. 启动服务：
   ```bash
   docker compose up -d
   ```
3. 查看运行日志：
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
