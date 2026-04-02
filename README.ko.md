<h1 align="center">cr-train</h1>

<p align="center">
  <em>위성 구름 제거를 위한 단일 클래스 학습 툴킷 -- 결정적 샘플링, 스마트 캐싱, DDP 기본 지원.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.4+"></a>
  <a href="https://huggingface.co/datasets/Hermanni/sen12mscr"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hermanni%2Fsen12mscr-yellow" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>한국어</b>
</p>

---

<details>
<summary><b>목차</b></summary>

- [주요 특징](#주요-특징)
- [빠른 시작](#빠른-시작)
- [예제](#예제)
- [Trainer가 자동으로 하는 일](#trainer가-자동으로-하는-일)
- [API 레퍼런스](#api-레퍼런스)
- [내부 구조](#내부-구조)
- [분산 학습](#분산-학습)
- [모델 규약](#모델-규약)
- [라이선스](#라이선스)

</details>

---

## 주요 특징

- **단일 클래스 API** -- `Trainer`의 `step()` + `test()`만으로 학습 완료
- **결정적 블록 샘플링** -- 단일 시드 시스템(`seed`)으로 완벽한 재현성
- **스마트 캐시 워밍업** -- HF streaming row-group block 중 누락된 블록만 채우고, 다른 계획에서도 재사용
- **분산 학습** -- 자동 DDP 래핑, rank-aware block 분할, all-reduce 메트릭
- **JSONL 실험 기록** -- 매 epoch, 검증, 체크포인트, 시작 이벤트를 `metrics.jsonl`에 기록
- **별도 데이터 다운로드 불필요** -- [`Hermanni/sen12mscr`](https://huggingface.co/datasets/Hermanni/sen12mscr)를 `load_dataset(..., streaming=True)`로 직접 사용

---

## 빠른 시작

### 설치

```bash
uv add git+https://github.com/smturtle2/cr-train.git
```

### 최소 예제

```python
from cr_train import Trainer
import torch
from torch import nn
from torch.nn import functional as F

class FusionBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # SAR 2채널 + 구름 낀 광학 13채널 = 총 15 입력 채널
        self.body = nn.Sequential(
            nn.Conv2d(15, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 13, 1),  # 타겟 광학 13채널
        )

    def forward(self, sar, cloudy):
        return self.body(torch.cat([sar, cloudy], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionBaseline().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trainer = Trainer(
    model, optimizer,
    loss=lambda pred, batch: F.l1_loss(pred, batch["target"]),
    metrics={"mae": lambda pred, batch: torch.mean(torch.abs(pred - batch["target"]))},
    max_train_samples=2048,
    max_val_samples=256,
    max_test_samples=256,
    batch_size=4,
    epochs=2,
    seed=42,
    output_dir="runs/sen12mscr",
)

for _ in range(trainer.epochs):
    print(trainer.step())

print(trainer.test())
```

<details>
<summary><b>예상 출력</b></summary>

```
train  ░░░░░…░░█░…░░░█░…  32 blocks (2 048 rows)
val    ░░░░░░░░░░░█░░░░░…   4 blocks (  244 rows)

Epoch 1/2 ━━━━━━━━━━━━━━━━━━━━ 512/512  loss=0.0423  mae=0.0312  12.3s
  val  loss=0.0391  mae=0.0298  ckpt=runs/sen12mscr/epoch-0001.pt

Epoch 2/2 ━━━━━━━━━━━━━━━━━━━━ 512/512  loss=0.0387  mae=0.0295  11.8s
  val  loss=0.0372  mae=0.0281  ckpt=runs/sen12mscr/epoch-0002.pt

Test  loss=0.0387  mae=0.0295  (256 samples)
```

</details>

---

## 예제

### CLI 학습

내장된 `FusionBaseline` 모델로 학습 스크립트를 실행합니다:

```bash
uv run python examples/train_sen12mscr.py \
  --max-train-samples 2048 \
  --max-val-samples 256 \
  --max-test-samples 256 \
  --batch-size 4 \
  --epochs 2 \
  --output-dir runs/sen12mscr-example
```

`--max-train-samples none` (또는 `full`)을 전달하면 전체 split을 캐싱하여 학습합니다.

### 샘플링 알고리즘 시각화

uniform exact-k 블록 선택 비트마스크가 어떻게 만들어지는지 확인합니다:

```bash
uv run python examples/bitmask_sampling_demo.py \
  --total-rows 107072 \
  --requested-rows 2048 \
  --seed 9
```

raw draw order, 최종 선택 블록 인덱스, 그리고 선택(`■`) vs. 건너뜀(`□`)의 logical block 비트맵을 출력합니다.

---

## Trainer가 자동으로 하는 일

대부분의 사용자는 `from cr_train import Trainer`만 쓰면 됩니다. `Trainer`를 만들면 다음을 자동으로 처리합니다:

- 데이터셋 메타데이터와 로컬 캐시 상태 확인
- 현재 호출에 필요한 split만 워밍업
- iterable DataLoader와 rank-aware block 분할 구성
- `metrics.jsonl`과 `epoch-NNNN.pt` 체크포인트 기록
- 학습 중 배치 단위 tqdm으로 running-average loss와 메트릭을 표시하고, epoch 종료 시 소요 시간을 출력
- `epoch-NNNN.pt` 체크포인트에 `model`, `optimizer`, `epoch`, `global_step` 상태 저장
- 출력 디렉토리의 `metrics.jsonl`에 줄 단위 JSON 객체로 메트릭 기록

일반적인 학습 흐름에서는 캐시나 데이터로더 헬퍼를 직접 호출할 필요가 없습니다.

---

## API 레퍼런스

### `Trainer.__init__`

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `model` | `nn.Module` | *(필수)* | PyTorch 모델. `forward(sar, cloudy)` 시그니처, 예측 텐서를 반환. |
| `optimizer` | `Optimizer` | *(필수)* | `model.parameters()`로 생성해야 함. |
| `loss` | `Callable` | *(필수)* | `(prediction, batch) -> 스칼라 텐서`. |
| `metrics` | `dict[str, Callable]` | `None` | `{"이름": (prediction, batch) -> 스칼라}`. epoch별 기록. |
| `max_train_samples` | `int \| None` | `None` | 요청 학습 행 수. 고정 `BLOCK_SIZE=64` 기준으로 블록 수로 변환. `None` = 전체 split. |
| `max_val_samples` | `int \| None` | `None` | 검증용 동일. |
| `max_test_samples` | `int \| None` | `None` | 테스트용 동일. |
| `batch_size` | `int` | `4` | 모든 DataLoader의 배치 크기. |
| `epochs` | `int` | `1` | 총 학습 epoch 수. epoch당 `step()` 한 번 호출. |
| `seed` | `int` | `42` | 결정적 블록 선택과 epoch별 block/row 셔플 순서를 제어하는 시드. |
| `output_dir` | `str \| Path` | `"runs/default"` | `metrics.jsonl` 및 체크포인트 파일 디렉토리. |
| `cache_dir` | `str \| Path \| None` | `None` | block 캐시 루트. `None` = `~/.cache/cr-train`. |

### `Trainer.step() -> dict`

학습 1 epoch + 검증 + 체크포인트를 실행합니다. 반환값:

```python
{
    "epoch": 1,
    "train": {
        "loss": 0.0423,
        "metrics": {"mae": 0.0312},
        "num_samples": 2048,
        "num_batches": 512,
        "samples_per_sec": 142.3,
        "batches_per_sec": 17.8,
    },
    "val": {
        "loss": 0.0391,
        "metrics": {"mae": 0.0298},
        "num_samples": 256,
        "num_batches": 64,
    },
    "checkpoint_path": "runs/sen12mscr/epoch-0001.pt",
    "elapsed_sec": 12.3,
}
```

### `Trainer.test() -> dict`

현재 모델 상태로 테스트를 실행합니다. 반환값:

```python
{
    "epoch": 2,
    "loss": 0.0387,
    "metrics": {"mae": 0.0295},
    "num_samples": 256,
    "num_batches": 64,
}
```

### 고급: 블록 플래너 직접 사용

결정적 uniform exact-k 블록 플래너를 직접 들여다보려면 `cr_train.data` 하위 API를 사용합니다:

```python
from cr_train.data import BLOCK_SIZE, trace_plan_sample
```

전체 시각화는 [`examples/bitmask_sampling_demo.py`](examples/bitmask_sampling_demo.py)를 참고하세요.

---

## 내부 구조

`Trainer`는 HuggingFace streaming으로 데이터를 읽고, row group 기준의 재사용 가능한 로컬 block cache를 유지하며, 시작 이벤트를 `metrics.jsonl`에 기록합니다. 같은 `seed`를 쓰면 uniform exact-k 기준의 논리 블록 선택은 유지되고, 학습 block/row 순서는 `seed + epoch_index`에 따라 epoch마다 바뀝니다.

워밍업은 호출에 필요한 split만 수행합니다. `step()`은 `train`과 `validation`, `test()`는 `test`를 준비하며, 선택된 row-group block이 이미 로컬에 있지 않을 때만 HuggingFace에서 누락된 블록을 가져옵니다.

- 캐시 워밍업 시 tqdm 프로그레스 바로 블록 다운로드를 표시하고, 완료 시 한 줄 `■/□` 블록 타임라인을 출력합니다.
- 동일한 `seed`는 같은 uniform exact-k 블록 선택을 유지하며, 학습 배치 순서는 `seed + epoch_index`로 epoch마다 변경됩니다.
- 완료된 캐시는 자동 삭제되지 않습니다. 디스크 공간 회수를 위해 캐시 디렉토리에서 직접 삭제하세요.

---

## 분산 학습

`torch.distributed`가 초기화되면 Trainer가 자동으로 모델을 `DistributedDataParallel`로 래핑합니다. 코드 변경이 필요 없습니다:

```bash
torchrun --nproc_per_node=2 examples/train_sen12mscr.py \
  --max-train-samples 4096 \
  --epochs 5
```

- 결정적 block 분할로 rank별 데이터 분할
- 모든 프로세스에 걸쳐 메트릭 all-reduce
- rank 0만 `metrics.jsonl`과 체크포인트 파일 기록
- 파일 락을 통한 캐시 워밍업 조율

---

## 모델 규약

모델의 `forward` 메서드는 **두 개의 위치 인자**를 받습니다:

| 인자 | Shape | Dtype | 설명 |
|------|-------|-------|------|
| `sar` | `[B, 2, 256, 256]` | `float32` | Sentinel-1 SAR 영상 (2채널) |
| `cloudy` | `[B, 13, 256, 256]` | `float32` | 구름 낀 Sentinel-2 광학 영상 (13채널) |

**출력**: 예측 텐서, 일반적으로 `[B, 13, 256, 256]`.

```python
class MyModel(nn.Module):
    def forward(self, sar, cloudy):
        # sar:    [B, 2,  256, 256]
        # cloudy: [B, 13, 256, 256]
        x = torch.cat([sar, cloudy], dim=1)  # [B, 15, 256, 256]
        return self.network(x)
```

**손실 함수**와 **메트릭 함수**는 `(prediction, batch)`를 입력받으며, `batch`는 `"sar"`, `"cloudy"`, `"target"`, `"meta"`를 포함하는 전체 dict입니다:

```python
def my_loss(prediction, batch):
    return F.l1_loss(prediction, batch["target"])
```

---

## 라이선스

[MIT](LICENSE)
