# llama.cpp TurboQuant + CUDA 13.3 (RTX 50 系列专用镜像)

本项目针对 **NVIDIA GeForce RTX 50 系列（Blackwell 架构，RTX 5090 / 5080 / 5070 / 5060）** 提供专用的 GitHub Actions 快速编译方案。

通过仅指定 **Blackwell `sm_120`** 单一计算架构，将 GitHub Actions 编译时间从 40+ 分钟大幅压缩至 **3~4 分钟**，并完整支持 **TurboQuant**（KV Cache 2~4bit 极低比特压缩）与 **CUDA 13.3** 特性。

---

## 🌟 核心参数与特性

- **专属显卡架构**：`CMAKE_CUDA_ARCHITECTURES=120`（专为 RTX 50 系列 Blackwell 核心定制，无冗余多架构开销）。
- **极速构建**：仅编译单架构，单次编译耗时仅需 **3~4 分钟**；结合 `ccache` 与 GitHub Actions `type=gha` 缓存，二次构建秒级完成。
- **极致显存节省**：集成 TurboQuant 算法，支持 `-ctk turbo4 -ctv turbo4`（或 `-ctk q8_0 -ctv turbo4`），长文本显存占用骤降 60%~75%。
- **CUDA 13.3 原生适配**：启用 FlashAttention (`-DGGML_CUDA_FA_ALL_QUANTS=ON`) 与 CUDA 13.3 深度优化。

---

## 📁 目录结构

```text
├── .github/
│   └── workflows/
│       └── build-and-push.yml   # 专为 50 系优化的 Actions 工作流
├── Dockerfile                   # 多阶段编译 Dockerfile (sm_120 + ccache)
├── docker-compose.yml           # RTX 50 系容器本地部署编排
├── .dockerignore                # 忽略构建无关文件
└── README.md                    # 说明文档
```

---

## 🚀 使用 GitHub Actions 一键构建

### 1. 开启仓库推送权限
1. 打开 GitHub 仓库页面，点击 **Settings** -> **Actions** -> **General**。
2. 滚动到底部 **Workflow permissions**，勾选 **Read and write permissions** 并保存。

### 2. 推送代码
```bash
git add .
git commit -m "feat: optimize build specifically for RTX 50 series (sm_120)"
git push
```

### 3. 触发构建
- 在仓库页面进入 **Actions** -> **Build llama.cpp TurboQuant CUDA 13.3 (RTX 50-Series Only)**。
- 点击 **Run workflow**（默认参数即为 `cuda_arch: 120` 和 `cuda_version: 13.3.0`），点击开始。
- **约 3~4 分钟即可构建完成！**

---

## 💻 运行与使用

### 方式一：Docker CLI 运行

```bash
docker run -d --name llama-server \
  --gpus all \
  -p 8080:8080 \
  -v /path/to/models:/models \
  ghcr.io/<你的用户名>/<仓库名>:rtx50 \
  -m /models/your-model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -fa 1 \
  -ctk turbo4 \
  -ctv turbo4 \
  -c 32768
```

### 方式二：Docker Compose 运行

```bash
docker compose up -d
```
