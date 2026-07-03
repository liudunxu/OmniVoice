# Vast.ai GPU 模型服务部署 Runbook

这份 runbook 是给同事或 coding agent 用的通用版：目标是把任意 GPU 模型服务打成 Docker 镜像，推到 Docker Hub，再从 Vast.ai 拉镜像启动服务。本文不绑定 LatentSync；LatentSync 只放在附录作为示例。

适用前提：

- 服务可以在 Linux 容器里运行。
- 服务监听 `0.0.0.0:<PORT>`，不要只监听 `127.0.0.1`。
- 镜像可以从 Docker Hub 拉取。
- 模型权重、API key、token 不写进镜像和 Git 历史。

## 1. 总流程

```text
GitHub repo
  -> Dockerfile
  -> GitHub Actions build
  -> push Docker Hub image
  -> Vast.ai search offer
  -> create instance with image/env/ports
  -> health check
  -> smoke test
  -> destroy instance when done
```

建议顺序：

1. 先在 GitHub Actions 里构建镜像并推到 Docker Hub。
2. 先用 US/CA 便宜 24GB 单卡验证镜像能启动。
3. smoke test 通过后，再挑 CN/目标地区机器。
4. 测完立刻销毁 Vast 实例。

## 2. Docker Hub 准备

### 2.1 创建 Docker Hub 账号

同事需要有一个 Docker Hub 账号，例如：

```text
username: your_dockerhub_username
```

登录 Docker Hub 网页后创建 repository：

```text
Repository name: your-model-service
Visibility: private 或 public
```

最终镜像名格式：

```text
your_dockerhub_username/your-model-service:vast-gpu
```

建议 tag：

```text
vast-gpu              # 浮动 tag，给 Vast.ai 默认使用
vast-gpu-<git-sha>   # 固定 tag，排查问题时使用
```

### 2.2 创建 Docker Hub Access Token

Docker Hub 网页：

```text
Account Settings
  -> Personal access tokens
  -> Generate new token
```

权限建议：

```text
Read, Write
```

保存 token 后不要发到聊天或提交到 repo。

## 3. GitHub Secrets 配置

在 GitHub 仓库页面：

```text
Settings
  -> Secrets and variables
  -> Actions
  -> New repository secret
```

至少添加：

```text
DOCKERHUB_USERNAME = your_dockerhub_username
DOCKERHUB_TOKEN    = Docker Hub access token
```

可选 secrets：

```text
HF_TOKEN           = Hugging Face token, 如果构建或启动时需要下载私有模型
GH_PAT             = GitHub personal access token, 如果容器启动时要 git pull 私有仓库
```

原则：

- GitHub Actions 构建镜像用 `DOCKERHUB_USERNAME` 和 `DOCKERHUB_TOKEN`。
- Vast.ai 运行时用 Vast 的 env 注入 `HF_TOKEN`、业务 API key 等。
- 不要把任何 token 写进 Dockerfile、README、`.env.example` 或镜像层。

## 4. Dockerfile 基础模板

不同模型服务可以替换 base image 和安装命令。GPU 推理常见基础镜像：

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    MODEL_DIR=/workspace/models \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY your_package/ /app/your_package/

RUN pip install --no-cache-dir -e /app

COPY docker/entrypoint.sh /usr/local/bin/model-service-entrypoint
RUN chmod +x /usr/local/bin/model-service-entrypoint

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/model-service-entrypoint"]
```

最小 entrypoint 示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

export PORT MODEL_DIR HF_HOME
mkdir -p "${MODEL_DIR}" "${HF_HOME}"

# 如果需要启动时下载模型，在这里处理；不要把 token 打印出来。
# python -m your_package.download_models --output "${MODEL_DIR}"

exec python -m your_package.server --host 0.0.0.0 --port "${PORT}"
```

服务必须监听：

```text
0.0.0.0:<PORT>
```

健康检查建议提供：

```text
GET /health
GET /v1/models
```

## 5. GitHub Actions 构建并推送 Docker Hub

创建 `.github/workflows/docker-hub-gpu.yml`：

```yaml
name: Build and Push GPU Image to Docker Hub

on:
  workflow_dispatch:
    inputs:
      image_name:
        description: Docker Hub image name, including namespace
        required: true
        default: your_dockerhub_username/your-model-service
      tag:
        description: Human-readable image tag
        required: true
        default: vast-gpu

jobs:
  docker-hub-gpu:
    runs-on: ubuntu-24.04
    permissions:
      contents: read
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Generate Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ inputs.image_name }}
          tags: |
            type=raw,value=${{ inputs.tag }}
            type=sha,prefix=${{ inputs.tag }}-

      - name: Build and push GPU image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile.gpu
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

手动触发：

```bash
gh workflow run docker-hub-gpu.yml \
  -f image_name=your_dockerhub_username/your-model-service \
  -f tag=vast-gpu
```

查看结果：

```bash
gh run list --workflow docker-hub-gpu.yml --limit 5
```

Docker Hub 上应该出现：

```text
your_dockerhub_username/your-model-service:vast-gpu
your_dockerhub_username/your-model-service:vast-gpu-<short-sha>
```

建议 Vast.ai 测试时优先用固定 sha tag，避免缓存或浮动 tag 混淆：

```text
your_dockerhub_username/your-model-service:vast-gpu-abc1234
```

## 6. Vast.ai API key 使用方式

本地 shell 设置：

```bash
export VAST_API_KEY='vast api key'
```

不要把 key 写进脚本文件。下面所有 Python 示例都从环境变量读取：

```python
api_key = os.environ["VAST_API_KEY"]
```

## 7. 下单前查余额和当前实例

```bash
python3 - <<'PY'
import json, os, urllib.parse, urllib.request

api_key = os.environ["VAST_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}"}

def get(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

user = get("https://console.vast.ai/api/v0/users/current/")
print("balance/credit:")
print(json.dumps({k: user.get(k) for k in ["balance", "credit"]}, indent=2))

cols = urllib.parse.quote(json.dumps([
    "id", "label", "actual_status", "cur_state", "gpu_name", "dph_total", "ports"
]))
instances = get(f"https://console.vast.ai/api/v1/instances/?select_cols={cols}&limit=50")
print("instances:")
print(json.dumps(instances.get("instances", []), indent=2))
PY
```

如果实例列表不为空，先判断是否需要保留。没用就销毁。

## 8. 搜便宜 GPU 实例

MVP 验证建议先搜 US/CA，镜像稳定后再看 CN：

```bash
python3 - <<'PY'
import json, os, urllib.request

api_key = os.environ["VAST_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

payload = {
    "limit": 200,
    "type": "ondemand",
    "verified": {"eq": True},
    "rentable": {"eq": True},
    "rented": {"eq": False},
    "num_gpus": {"eq": 1},
    "gpu_ram": {"gte": 24000},
    "disk_space": {"gte": 50},
    "direct_port_count": {"gte": 1},
    "dph_total": {"lte": 0.25}
}

req = urllib.request.Request(
    "https://console.vast.ai/api/v0/bundles/",
    data=json.dumps(payload).encode(),
    headers=headers,
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read().decode())

offers = []
for offer in data.get("offers", []):
    geo = str(offer.get("geolocation") or "")
    country = str(offer.get("country_code") or "").upper()
    if country in {"US", "CA"} or geo.strip().endswith(("US", "CA")):
        offers.append(offer)

offers.sort(key=lambda o: (float(o.get("dph_total") or 999), -float(o.get("reliability") or 0)))
for offer in offers[:10]:
    print(json.dumps({
        "id": offer.get("id"),
        "machine_id": offer.get("machine_id"),
        "gpu_name": offer.get("gpu_name"),
        "gpu_ram": offer.get("gpu_ram"),
        "geolocation": offer.get("geolocation"),
        "dph_total": offer.get("dph_total"),
        "disk_space": offer.get("disk_space"),
        "inet_down": offer.get("inet_down"),
        "inet_up": offer.get("inet_up"),
        "disk_bw": offer.get("disk_bw"),
        "reliability": offer.get("reliability"),
    }, ensure_ascii=False))
PY
```

挑选建议：

- 先用 RTX 3090 24GB 做 API/MVP 验证，通常便宜。
- 真正生产再换 RTX 4090 或更合适的卡。
- 需要下载大模型时，优先看 `inet_down`。
- 需要频繁上传/下载视频时，也看 `inet_up`。
- `disk_space` 是可租磁盘余量；创建时仍要显式指定 `disk`。
- 如果价格接近，优先选 `reliability` 更高、`disk_bw` 更高的机器。

## 9. 创建 Vast.ai 实例

推荐：

```text
runtype=args
```

原因：`args` 会让 Docker 镜像自己的 `ENTRYPOINT` 成为主进程。我们实际踩坑发现，`ssh_direct` 会改写镜像入口，容易导致服务没启动或 onstart 行为不稳定。

通用创建示例：

```bash
python3 - <<'PY'
import json, os, urllib.request

api_key = os.environ["VAST_API_KEY"]
offer_id = 41028912  # 替换为搜索到的 offer id

payload = {
    "image": "your_dockerhub_username/your-model-service:vast-gpu-abc1234",
    "disk": 50,
    "runtype": "args",
    "target_state": "running",
    "cancel_unavail": True,
    "label": "your-model-service-mvp",
    "env": {
        "-p 8000:8000": "1",
        "PORT": "8000",
        "MODEL_DIR": "/workspace/models",
        "HF_HOME": "/workspace/.cache/huggingface"
    }
}

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
req = urllib.request.Request(
    f"https://console.vast.ai/api/v0/asks/{offer_id}/",
    data=json.dumps(payload).encode(),
    headers=headers,
    method="PUT",
)
with urllib.request.urlopen(req, timeout=90) as resp:
    print(resp.read().decode())
PY
```

成功响应包含：

```json
{
  "success": true,
  "new_contract": 12345678
}
```

记录 `new_contract`。拿到它以后，就按可能正在计费处理。

### 常见 env 写法

端口映射：

```json
"-p 8000:8000": "1"
```

Hugging Face 缓存：

```json
"HF_HOME": "/workspace/.cache/huggingface"
```

模型目录：

```json
"MODEL_DIR": "/workspace/models"
```

私有模型 token：

```json
"HF_TOKEN": "通过安全方式注入，不写进仓库"
```

## 10. 等实例启动并拿公网端口

```bash
python3 - <<'PY'
import json, os, time, urllib.parse, urllib.request

api_key = os.environ["VAST_API_KEY"]
instance_id = 12345678
headers = {"Authorization": f"Bearer {api_key}"}

for _ in range(60):
    filters = urllib.parse.quote(json.dumps({"id": {"eq": instance_id}}))
    cols = urllib.parse.quote(json.dumps([
        "id", "actual_status", "cur_state", "status_msg", "ports",
        "public_ipaddr", "dph_total", "gpu_name", "disk_usage"
    ]))
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v1/instances/?select_filters={filters}&select_cols={cols}&limit=5",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    inst = (body.get("instances") or [{}])[0]
    print(json.dumps(inst, ensure_ascii=False))
    if inst.get("actual_status") == "running" and inst.get("ports"):
        break
    if inst.get("cur_state") == "stopped" and "Error:" in str(inst.get("status_msg")):
        break
    time.sleep(20)
PY
```

端口返回示例：

```json
{
  "public_ipaddr": "137.175.76.24",
  "ports": {
    "8000/tcp": [{"HostPort": "11200"}]
  }
}
```

公网服务地址：

```text
http://137.175.76.24:11200
```

## 11. 健康检查和日志

通用健康检查：

```bash
curl -fsS http://<public_ip>:<public_port>/health
```

如果你的服务没有 `/health`，至少准备一个轻量接口，例如：

```text
GET /
GET /v1/models
GET /api/jobs
```

如果返回 `Empty reply from server`：

- 端口已经通了。
- 容器里的服务还没有 ready。
- 常见原因：正在下载模型权重、初始化 CUDA、加载模型。

抓 Vast 容器日志：

```bash
python3 - <<'PY'
import json, os, time, urllib.request

api_key = os.environ["VAST_API_KEY"]
instance_id = 12345678
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

req = urllib.request.Request(
    f"https://console.vast.ai/api/v0/instances/request_logs/{instance_id}",
    data=json.dumps({"tail": "800"}).encode(),
    headers=headers,
    method="PUT",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.loads(resp.read().decode())

time.sleep(5)
with urllib.request.urlopen(body["result_url"], timeout=30) as resp:
    print(resp.read().decode(errors="replace")[-12000:])
PY
```

建议服务自身把关键日志写到 stdout/stderr，这样 Vast 日志能直接看到：

```text
model download start/done
server bind host/port
model loaded
first request started/done
error traceback
```

## 12. Smoke Test 判断标准

不要只看接口返回 200。至少检查：

- 健康检查成功。
- 一个最小输入能跑通。
- 输出文件或 JSON 结构符合预期。
- 日志里没有 fallback/error。
- GPU 模型确实被调用，而不是静默跳过。

通用 smoke 示例：

```bash
curl -fsS -X POST \
  -F input=@/path/to/small_input.file \
  http://<public_ip>:<public_port>/predict
```

如果是 OpenAI-compatible 模型服务：

```bash
curl -fsS http://<public_ip>:<public_port>/v1/models
```

## 13. 销毁实例止损

测试完立刻销毁：

```bash
python3 - <<'PY'
import os, urllib.request

api_key = os.environ["VAST_API_KEY"]
instance_id = 12345678
req = urllib.request.Request(
    f"https://console.vast.ai/api/v0/instances/{instance_id}/",
    headers={"Authorization": f"Bearer {api_key}"},
    method="DELETE",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    print(resp.status, resp.read().decode())
PY
```

销毁后再查一次实例列表和 credit，确认没有实例残留。

## 14. 常见坑

- `ssh_direct` 会改写 entrypoint。服务型容器优先用 `runtype=args`。
- 创建失败且没有 `new_contract` 通常没有实例；有 `new_contract` 后就要按可能计费处理。
- Docker Hub 浮动 tag 可能被缓存或混淆；排查问题用固定 sha tag。
- CN 机房拉 Docker Hub / Hugging Face 可能慢；镜像稳定前用 US/CA 更省时间。
- 私有 GitHub 仓库不能无 token `git clone`；没有 token 时不要开启启动时 auto update。
- 服务必须监听 `0.0.0.0`。
- 模型权重不要 bake 到 Git；要么启动时下载到 `/workspace`，要么用受控镜像层/volume。
- 任务返回 `completed` 不一定代表模型成功；要检查输出内容和错误字段。
- 日志必须足够详细，否则远端调试会非常浪费 GPU 时间。

## 15. 附录：当前 LatentSync Worker 示例

当前镜像：

```text
bigmy1/ai-dubbing-lipsync:vast-gpu-fa71415
```

Vast env：

```json
{
  "-p 6006:6006": "1",
  "LATENTSYNC_PORT": "6008",
  "WORKER_PORT": "6006",
  "REMOTE_JOB_ROOT": "/workspace/remote_jobs",
  "LATENTSYNC_TEMP_DIR": "/workspace/latentsync_temp",
  "LATENTSYNC_CHECKPOINT_PATH": "/workspace/checkpoints/latentsync_unet.pt",
  "HF_HOME": "/workspace/.cache/huggingface",
  "LATENTSYNC_DOWNLOAD_CHECKPOINTS": "1",
  "AI_DUBBING_AUTO_UPDATE": "0"
}
```

健康检查：

```bash
curl -fsS http://<public_ip>:<worker_port>/api/jobs
```

提交任务：

```bash
curl -fsS \
  -F task_id=smoke_g1p5_s20_001 \
  -F video=@/path/to/input.mp4 \
  -F guidance_scale=1.5 \
  -F inference_steps=20 \
  -F roi_size=640 \
  http://<public_ip>:<worker_port>/api/jobs
```

下载 manifest：

```bash
curl -fsS -L -o processed_manifest.json \
  http://<public_ip>:<worker_port>/download/smoke_g1p5_s20_001/processed_manifest
```

检查是否真跑了模型：

```bash
python3 - <<'PY'
import collections, json
data = json.load(open("processed_manifest.json"))
segments = data.get("segments", [])
print(collections.Counter(s.get("action") for s in segments))
for item in segments:
    if item.get("error"):
        print(item.get("id"), item.get("action"), item.get("error"))
PY
```

当前已知状态：

```text
已修复: checkpoint symlink
仍需修复: Gradio show_error/内部日志暴露；部分片段 Face not detected 或 Gradio 内部异常
```
