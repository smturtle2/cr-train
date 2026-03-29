# cr-train

[English](./README.md) | 한국어

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

[`Hermanni/sen12mscr`](https://huggingface.co/datasets/Hermanni/sen12mscr)를 대상으로, scene-level split과 재현 가능한 batch를 전제로 만든 모델-불가지론적 streaming 학습 모듈입니다.

## 왜 이 프로젝트가 필요한가

Hugging Face에 올라온 SEN12MS-CR 미러는 편하지만, 실제 학습에 바로 쓰기에는 몇 가지가 부족합니다.

- `sar`, `cloudy`, `target`이 raw bytes로 저장되어 있음
- 공식 `train / val / test` split이 바로 노출되지 않음
- scene 단위 누수 없는 평가 구성이 필요함
- streaming-only 경로에서 재현 가능한 batch가 필요함
- 특정 모델, loss, optimizer, scheduler에 종속되지 않는 학습 루프가 필요함

`cr-train`은 이 부분을 데이터 계층과 학습 계층으로 나눠 정리한 모듈입니다.

## 핵심 특징

- 항상 streaming으로만 동작합니다.
- split은 `season/scene parquet shard` 단위로 분리됩니다.
- 공식 split manifest를 번들로 포함합니다.
- `sar`, `cloudy`, `target` bytes는 커스텀 dataset 내부에서 tensor로 복원됩니다.
- train shuffle은 항상 `reshard() -> shuffle(seed, buffer_size)` 순서를 지킵니다.
- 같은 seed, split, loader topology면 같은 batch가 나오도록 기본값을 보수적으로 잡았습니다.
- `max_batches`로 partial epoch 학습과 평가를 지원합니다.
- `Trainer`는 model, optimizer, `step_fn`, optional scheduler를 외부에서 주입받습니다.
- 모델 device는 호출자가 명시적으로 소유합니다.

## 설치

```bash
uv sync
```

Hugging Face rate limit을 줄이려면 토큰 설정을 권장합니다.

```bash
export HF_TOKEN=your_token
```

## 빠른 시작

```python
import torch
import torch.nn.functional as F
from torch import nn

from cr_train import (
    DataModuleConfig,
    LoaderConfig,
    SEN12MSCRDataModule,
    ShuffleConfig,
    StepResult,
    Trainer,
)


class TinyCloudRemovalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 13, kernel_size=1),
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([sar.float(), cloudy.float()], dim=1))


def step_fn(model: nn.Module, batch: dict[str, torch.Tensor], stage: str) -> StepResult:
    prediction = model(batch["sar"], batch["cloudy"])
    target = batch["target"].float()
    loss = F.mse_loss(prediction, target)
    return StepResult(loss=loss, metrics={"mae": F.l1_loss(prediction, target)})


datamodule = SEN12MSCRDataModule(
    DataModuleConfig(
        split_strategy="seeded_scene",
        seed=7,
        loader=LoaderConfig(batch_size=4),
        shuffle=ShuffleConfig(buffer_size=16, reshard_num_shards=16),
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCloudRemovalNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    datamodule=datamodule,
    step_fn=step_fn,
    checkpoint_dir="artifacts/checkpoints",
)

history = trainer.fit(max_epochs=2, train_max_batches=10, val_max_batches=2)
test_metrics = trainer.test(test_max_batches=2)
print(history)
print(test_metrics)
```

예제 실행:

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 8 --val-max-batches 2
```

이 프로젝트의 `Trainer` 계약은 다음 순서를 전제로 합니다.

- 먼저 model을 최종 device로 옮깁니다.
- 그 model 기준으로 optimizer를 생성합니다.
- 둘 다 `Trainer`에 넘깁니다.

## 데이터 동작 보장

### 1. Streaming only

map-style preprocessing 계층을 두지 않습니다. Hugging Face iterable streaming 경로를 그대로 사용합니다.

### 2. Scene-isolated split

기본 custom split 전략은 `seeded_scene`입니다. `reshard()` 이후에 split하지 않으므로, 하나의 원본 `season/scene parquet shard`는 정확히 하나의 split에만 속합니다.

### 3. Bytes -> tensor 복원

각 sample은 dataset 내부에서 다음 규칙으로 복원됩니다.

- `sar`: `sample["dtype"]`, `sample["sar_shape"]` 사용
- `cloudy`: `int16`, `sample["opt_shape"]` 사용
- `target`: `int16`, `sample["opt_shape"]` 사용

기본 tensor layout은 `channels_first`입니다.

### 4. Shuffle invariant

train shuffle 경로는 항상 아래 순서를 따릅니다.

```text
scene split -> streaming dataset -> reshard() -> shuffle(seed, buffer_size) -> batch
```

scene split 안전성을 유지하면서도 학습 시 row-group mixing을 확보하기 위한 고정 규칙입니다.

### 5. Reproducibility

기본 loader 설정은 재현성을 우선합니다.

- `num_workers=0`
- `in_order=True`
- `persistent_workers=False`

같은 seed, split strategy, batch size, loader topology면 같은 batch가 나옵니다.  
worker 수나 prefetch를 늘리면 처리량 위주의 모드로 보는 것이 맞습니다.

## Split 전략

### `official`

저자들이 제공한 supplementary split metadata를 정규화한 scene-level manifest를 사용합니다. 번들 파일은 [`src/cr_train/resources/official_scene_splits.csv`](./src/cr_train/resources/official_scene_splits.csv)입니다.

원본 출처:

- 공식 페이지: <https://patricktum.github.io/cloud_removal/sen12mscr/>
- supplementary 폴더: <https://u.pcloud.link/publink/show?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV>
- 현재 split 파일: <https://api.pcloud.com/getpubtextfile?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV&fileid=57823192235>

### `seeded_scene`

전체 scene catalog를 season-stratified 방식으로 섞어 deterministic custom split을 만듭니다.

기본 비율:

```python
SplitRatios(train=0.8, val=0.1, test=0.1)
```

기본 shuffle 설정:

```python
ShuffleConfig(buffer_size=16, reshard_num_shards=1024)
```

SEN12MS-CR sample은 크기가 커서 streaming buffer를 크게 잡으면 host memory를 빠르게 많이 씁니다.

## API 요약

```python
from cr_train import (
    DataModuleConfig,
    LoaderConfig,
    SEN12MSCRDataModule,
    ShuffleConfig,
    SplitRatios,
    StepResult,
    Trainer,
)
```

주요 설정 포인트:

- `DataModuleConfig.split_strategy`: `official` 또는 `seeded_scene`
- `DataModuleConfig.seed`: split과 shuffle에 공통으로 쓰는 seed
- `ShuffleConfig.buffer_size`: streaming shuffle buffer 크기
- `ShuffleConfig.reshard_num_shards`: shuffle 전 reshard 목표값
- `LoaderConfig.batch_size`: batch size
- `Trainer`: model이 이미 최종 device에 올라가 있다고 가정함
- `Trainer(..., scheduler=..., scheduler_step_fn=...)`: epoch 단위 scheduler hook 지원, metrics key는 `train/...`, `val/...`
- `Trainer.fit(train_max_batches=..., val_max_batches=...)`: partial epoch 학습
- `Trainer.test(test_max_batches=...)`: capped evaluation

Runtime 참고:

- Hugging Face parquet streaming bootstrap은 dataset loader 내부에서 자동으로 처리됩니다.
- 종료 시 crash를 피하기 위한 workaround는 내부 구현이며, 별도 public setup 단계는 없습니다.

## 프로젝트 구조

```text
src/cr_train/data.py                    # streaming dataset, split logic, dataloaders
src/cr_train/trainer.py                 # model-agnostic trainer
src/cr_train/resources/official_scene_splits.csv
examples/minimal_train.py               # minimal end-to-end usage
scripts/refresh_official_scene_splits.py
tests/                                  # reproducibility and trainer tests
```

## 개발

테스트 의존성 설치:

```bash
uv sync --dev
```

테스트 실행:

```bash
uv run pytest
```

공식 scene manifest 갱신:

```bash
uv run python scripts/refresh_official_scene_splits.py
```

## 상태

이 저장소는 의도적으로 좁은 범위를 다룹니다. SEN12MS-CR용 학습 모듈이지, 모델 zoo가 아닙니다. 추상화 경계는 네트워크 구조가 아니라 데이터 파이프라인과 학습 루프입니다.
