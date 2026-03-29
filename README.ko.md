# cr-train

[English](./README.md) | 한국어

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

[SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/) 구름 제거 실험을 위한 드롭인 학습 툴킷입니다.
PyTorch 모델만 가져오세요 -- 결정론적 스트리밍, 전처리, 체크포인팅, 진행 표시는 cr-train이 처리합니다.

---

## 설치

```bash
# 권장
uv add git+https://github.com/smturtle2/cr-train.git

# 또는 pip
pip install git+https://github.com/smturtle2/cr-train.git
```

> Hugging Face rate limit가 적용됩니다. `export HF_TOKEN=your_token`을 설정하면 더 빠릅니다.

## 빠른 시작

```python
import torch
from torch import nn
from cr_train import MAE, Trainer, TrainerConfig, build_sen12mscr_loaders


# 1. 모델 정의 -- (sar, cloudy) 두 개의 위치 인자를 받으면 됩니다
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(15, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 13, 1),
        )

    def forward(self, sar, cloudy):
        return self.net(torch.cat([sar, cloudy], dim=1))


# 2. 데이터 로더 생성 (Hugging Face에서 스트리밍, 로컬 다운로드 불필요)
train_loader, val_loader, test_loader = build_sen12mscr_loaders(batch_size=4)

# 3. 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)

trainer = Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    criterion=nn.MSELoss(),
    metrics=[MAE()],
    config=TrainerConfig(max_epochs=5, checkpoint_dir="checkpoints"),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

for epoch in trainer.step():
    print(f"Epoch {epoch['epoch']}  "
          f"train_loss={epoch['train']['loss']:.4f}  "
          f"val_loss={epoch['val']['loss']:.4f}")

print(trainer.test())
```

```bash
uv run my_train.py
```

더 많은 예제: [`examples/minimal_train.py`](./examples/minimal_train.py) (전체 옵션 CLI), [`examples/colab_quickstart.ipynb`](./examples/colab_quickstart.ipynb) (Colab 노트북).

---

## 동작 방식

```
Hugging Face (parquet shard)
  │
  ▼
build_sen12mscr_loaders()          ← scene 단위 분할, 스트리밍 디코딩, 전처리
  │
  ├── train_loader                 ← reshard + epoch별 shuffle
  ├── val_loader                   ← 고정 순서
  └── test_loader                  ← 고정 순서
        │
        ▼
    Trainer.step()                 ← train/val 루프, 진행 표시줄, 자동 체크포인팅
        │
        ▼
    Trainer.test()                 ← 최종 평가
```

**데이터 파이프라인.** Parquet shard를 Hugging Face에서 직접 스트리밍합니다 -- 로컬 다운로드가 필요 없습니다. 각 sample은 CHW 텐서(2x256x256 SAR, 13x256x256 optical)로 디코딩되고 즉시 정규화됩니다.

**Scene 격리.** 셔플링 전에 scene을 train/val/test에 배정합니다. 하나의 scene이 여러 split에 나타나지 않습니다.

**결정론적 순서.** 같은 `seed`를 사용하면 배치 순서가 완전히 재현됩니다. Trainer는 매 epoch마다 `set_epoch()`을 호출하여 셔플 순서가 결정론적으로 변경됩니다.

**체크포인팅.** `checkpoint_dir`을 설정하면 매 epoch 후 `last.pt`와 `epoch-NNNN.pt`가 자동 저장됩니다. 체크포인트에는 모델, optimizer, scheduler(설정 시), 전체 RNG 상태가 포함되어 정확한 재개가 가능합니다.

---

## 배치 형식

모든 배치는 다음 구조의 딕셔너리입니다:

```python
{
    "inputs": [
        Tensor,  # [B, 2, 256, 256]   SAR 후방산란, float32 [0, 1]
        Tensor,  # [B, 13, 256, 256]  구름 낀 Sentinel-2, float32 [0, 1]
    ],
    "target": Tensor,      # [B, 13, 256, 256]  구름 없는 Sentinel-2, float32 [0, 1]
    "metadata": {
        "season": list[str],        # "spring" | "summer" | "fall" | "winter"
        "scene":  list[str],        # scene ID
        "patch":  list[str],        # scene 내 patch ID
        "source_shard": list[str],  # 예: "spring/scene_1.parquet"
    },
}
```

`inputs`는 `[sar, cloudy]` 리스트입니다. Trainer는 이를 `model(sar, cloudy)`로 풀어서 호출합니다. `forward()`의 파라미터 이름은 자유입니다.

**전처리:**

| 밴드 | 원본 범위 | 전처리 | 출력 |
|------|-----------|--------|------|
| Optical (cloudy, target) | int16 [0, 10000] | `clip(0, 10000) / 10000` | float32 [0, 1] |
| SAR | float32 [-25, 0] dB | `clip(-25, 0)` 후 `(x + 25) / 25` | float32 [0, 1] |

---

## API 레퍼런스

### `build_sen12mscr_loaders(batch_size, **kwargs)`

`(train_loader, val_loader, test_loader)`를 반환합니다. train만 셔플되며 val/test는 고정 순서입니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `batch_size` | `int` | 필수 | 배치당 샘플 수 |
| `seed` | `int` | `0` | shuffle 순서 및 worker 초기화 시드 |
| `split` | `str` | `"official"` | `"official"` (저자 split: 155/10/10 scenes) 또는 `"seeded_scene"` (season별 계층화 80/10/10) |
| `shuffle_buffer_size` | `int` | `16` | 학습용 메모리 내 shuffle 버퍼 크기 |
| `num_workers` | `int \| None` | `None` | 자동: train은 `min(2, cpu_count//6)` workers, val/test는 0 |
| `pin_memory` | `bool` | `False` | 빠른 GPU 전송을 위해 텐서를 고정 메모리에 할당 |
| `prefetch_factor` | `int \| None` | `None` | workers > 0일 때 자동 `2` |
| `persistent_workers` | `bool \| None` | `None` | 자동 `True`; epoch 간 worker를 유지하여 재연결 오버헤드 방지 |
| `io_profile` | `str` | `"smooth"` | `"smooth"` (가벼운 readahead) 또는 `"conservative"` (완전 동기식) |

### `TrainerConfig`

불변(`frozen=True`) 학습 루프 설정.

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `max_epochs` | `int` | `1` | 학습 epoch 수 |
| `train_max_batches` | `int \| None` | `None` | epoch당 학습 배치 수 상한 |
| `val_max_batches` | `int \| None` | `None` | epoch당 검증 배치 수 상한 |
| `test_max_batches` | `int \| None` | `None` | 테스트 배치 수 상한 |
| `checkpoint_dir` | `str \| Path \| None` | `None` | 체크포인트 디렉토리 (자동 생성); `None`이면 저장 비활성화 |
| `show_progress` | `bool \| None` | `None` | 진행 표시줄; `None` = `True` |

### `Trainer`

```python
Trainer(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: (outputs, target) -> Tensor,
    metrics: [MAE(), ...] | None,
    config: TrainerConfig,
    train_loader,
    val_loader=None,
    test_loader=None,
    scheduler=None,            # 예: torch.optim.lr_scheduler.*
)
```

모델 파라미터에서 device를 자동 추론합니다. 배치 데이터는 해당 device로 자동 이동됩니다. Metric은 per-sample 기준으로 평균하므로 마지막 작은 배치가 결과를 왜곡하지 않습니다.

#### `trainer.step(**overrides) -> Iterator[dict]`

train/val epoch을 실행합니다. epoch당 하나의 레코드를 yield합니다:

```python
{
    "epoch": 1,           # 1부터 시작
    "global_step": 120,   # 누적 optimizer step 수
    "train": {"loss": 0.0512, "mae": 0.0321},
    "val":   {"loss": 0.0498, "mae": 0.0315},  # val_loader 없으면 {}
}
```

`scheduler`가 설정되어 있으면 매 epoch 후 `scheduler.step()`이 호출됩니다. 선택적 override: `max_epochs`, `train_max_batches`, `val_max_batches`.

#### `trainer.test(**overrides) -> dict[str, float]`

`{"loss": ..., "mae": ...}`를 반환합니다. 선택적 override: `max_batches`.

#### `trainer.save_checkpoint(filename) -> Path | None`

`checkpoint_dir/filename`에 저장합니다. 체크포인팅이 비활성화되어 있으면 `None` 반환.

#### `trainer.load_checkpoint(path)`

체크포인트 파일에서 전체 상태를 복원하여 정확한 학습 재개를 지원합니다.

### `MAE`

내장 L1 loss metric. 사용법: `metrics=[MAE()]`.

---

## 재현성

| 보장 사항 | 메커니즘 |
|-----------|----------|
| Scene 누수 없음 | 셔플링 전 scene 단위 분할 |
| 결정론적 배치 | 같은 `seed` + 같은 설정 = 동일한 순서 |
| Epoch별 셔플 | `set_epoch()` 자동 전파 (train만) |
| 정확한 재개 | 체크포인트에 모델, optimizer, scheduler, 전체 RNG 상태 캡처 |
| 고정된 데이터셋 버전 | Dataset revision 하드코딩 |

**Official split**은 [저자의 supplementary material](https://patricktum.github.io/cloud_removal/sen12mscr/)에서 가져왔으며 [`official_scene_splits.csv`](./src/cr_train/resources/official_scene_splits.csv)에 번들되어 있습니다.

---

## 프로젝트 구조

```
cr-train/
├── src/cr_train/
│   ├── __init__.py         # 공개 API
│   ├── trainer.py          # 학습 루프, 체크포인팅, 진행 표시
│   ├── data.py             # 스트리밍 데이터셋, scene 분할, 전처리
│   └── runtime.py          # parquet I/O 튜닝
├── examples/
│   ├── minimal_train.py    # 전체 옵션 CLI 학습 스크립트
│   └── colab_quickstart.ipynb
├── tests/
├── scripts/
│   └── refresh_official_scene_splits.py
└── pyproject.toml
```

## 라이선스

[MIT](./LICENSE)
