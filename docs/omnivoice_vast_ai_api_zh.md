# OmniVoice API Vast.ai 部署说明

本文是 `docs/vast_ai_gpu_service_runbook_zh.md` 的 OmniVoice 专用落地版。目标是在 Vast.ai 上运行 `api.py`，对外暴露 HTTP API。

同一个 `api.py` 进程同时提供：

- `POST /api/synthesize` / `POST /api/voxcpm/synthesize`：OmniVoice TTS
- `POST /api/separate`：人声/背景音分离，使用 Audio Separator / RoFormer 的 `vocals_mel_band_roformer.ckpt`
- `POST /api/audio_qc/reference`：reference 响度、多人、性别和可选跨 stem 音乐泄漏检查
- `POST /api/speaker/compare`：两个短语音片段的音色相似度和性别比较。默认使用 WavLM speaker embedding（`backend="wavlm_base_sv"`，首次请求时懒加载 `microsoft/wavlm-base-plus-sv`，可用 `SPEAKER_EMBED_MODEL` / `SPEAKER_EMBED_DEVICE` 调整）；加载失败或设 `SPEAKER_COMPARE_BACKEND=mfcc` 时回落 MFCC 启发式（`backend="mfcc_v1"`）。响应字段不变，`backend` 标识实际使用的实现

`docker/Dockerfile.vast` 构建阶段会同时预下载 OmniVoice TTS 权重和默认分离模型。启动后如果 `/workspace/models/audio-separator` 已经有 `vocals_mel_band_roformer.ckpt` 及其 yaml 配置，运行时不会再访问网络下载分离模型。

## 镜像选择

默认使用内置权重镜像：

```text
docker/Dockerfile.vast
```

它会在构建阶段把 `k2-fsa/OmniVoice` 下载到：

```text
/opt/omnivoice/models/k2-fsa/OmniVoice
```

容器启动时 `docker/entrypoint.sh` 会优先使用这个本地模型路径，避免 Vast.ai 实例每次启动后再从 Hugging Face 下载权重。

如果只想构建轻量镜像，不内置权重，可改用：

```text
docker/Dockerfile.gpu
```

## GitHub Actions 构建

先在 GitHub Secrets 配置：

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

如果模型需要 Hugging Face token，再加：

```text
HF_TOKEN
```

手动触发 workflow：

```bash
gh workflow run docker-hub-gpu.yml \
  -f image_name=your_dockerhub_username/omnivoice-api \
  -f tag=vast-gpu \
  -f dockerfile=docker/Dockerfile.vast
```

建议 Vast.ai 使用固定 sha tag，例如：

```text
your_dockerhub_username/omnivoice-api:vast-gpu-abc1234
```

## Vast.ai 创建实例参数

推荐使用：

```text
runtype=args
```

端口映射：

```json
"-p 8000:8000": "1"
```

推荐 env：

```json
{
  "-p 8000:8000": "1",
  "PORT": "8000",
  "HOST": "0.0.0.0",
  "MODEL_DIR": "/workspace/models",
  "AUDIO_SEPARATOR_MODEL_DIR": "/workspace/models/audio-separator",
  "OMNIVOICE_MAX_REQUEST_MB": "512"
}
```

不要覆盖 `HF_HOME`，除非你明确不想用镜像内置模型。内置权重镜像默认使用：

```text
HF_HOME=/opt/omnivoice/hf
```

## 服务接口

容器内服务启动命令等价于：

```bash
python /app/api.py --model /opt/omnivoice/models/k2-fsa/OmniVoice --ip 0.0.0.0 --port 8000
```

健康检查：

```bash
curl -fsS http://<public_ip>:<public_port>/health
curl -fsS http://<public_ip>:<public_port>/api/health
```

模型状态：

```bash
curl -fsS http://<public_ip>:<public_port>/api/status
curl -fsS http://<public_ip>:<public_port>/v1/models
```

合成接口：

```text
POST /api/voxcpm/synthesize
POST /api/synthesize
```

最小 smoke test：

```bash
curl -fsS -X POST "http://<public_ip>:<public_port>/api/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is OmniVoice running on Vast.ai.","language":"en"}'
```

首次合成会加载模型到 GPU，耗时会明显高于健康检查。

`/api/synthesize` 可传 `declared_gender` 和 `enable_speaker_check=true`。返回的
`audio_qc` 包含 `speaker_identity`、`speaker_similarity_to_reference`、
`prosody_match` 和 `emotion_similarity_to_reference`；高置信度男女声冲突会增加
`gender_mismatch` quality issue，交由调用方换 seed/reference 或保留原声。

VoxCPM 路径（`/api/voxcpm/synthesize`）现在也会在有 reference/prompt 音频时产出同样的
identity/prosody QC 字段（含 `gender_mismatch`、`emotion_mismatch` quality issue），
可用环境变量 `VOXCPM_IDENTITY_QC=0` 关闭；整个 QC fail-open，异常只记日志不影响合成。
