# cr-train

[English](./README.md) | 한국어

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

[`Hermanni/sen12mscr`](https://huggingface.co/datasets/Hermanni/sen12mscr)를 대상으로, deterministic scene-level split과 재현 가능한 batch를 전제로 만든 모델-불가지론적 streaming 학습 인프라입니다.

## 변경된 점

이 README는 현재의 breaking API를 기준으로 작성되었습니다.

- `DataModule`이 없습니다
- `step_fn`이 없습니다
- `Trainer`에 scheduler hook이 없습니다
- 데이터는 `build_sen12mscr_dataset(...)`, `build_sen12mscr_dataloader(...)`로 직접 만듭니다
- 학습은 `for history in trainer.step(): ...`로 실행합니다

## 왜 이 프로젝트가 필요한가

Hugging Face의 SEN12MS-CR 미러는 편리하지만, 실제 학습에 바로 쓰기에는 몇 가지가 더 필요합니다.

- end-to-end streaming-only 접근
- deterministic `train / val / test` split
- scene 단위 누수 없는 평가
- 내부 bytes-to-tensor decode
- 재현 가능한 batching
- 모델 구조와 loss 설계에 종속되지 않는 trainer

`cr-train`은 이 조합을 하나의 재사용 가능한 Python 모듈로 정리합니다.

## 핵심 특징

- 항상 streaming으로만 동작합니다.
- split은 `season/scene parquet shard` 단위로 분리됩니다.
- 공식 split manifest를 번들로 포함합니다.
- `sar`, `cloudy`, `target` bytes는 dataset 내부에서 tensor로 복원됩니다.
- 모든 sample과 batch는 `inputs / target / metadata` 스키마를 따릅니다.
- train shuffle은 항상 `reshard() -> shuffle(seed, buffer_size)` 순서를 지킵니다.
- 같은 seed, split strategy, loader topology면 같은 batch가 나오도록 기본값을 보수적으로 잡았습니다.
- `TrainerConfig`로 partial epoch 학습과 평가를 제어할 수 있습니다.
- 모델은 optimizer를 만들기 전에 먼저 최종 device로 옮겨야 합니다.

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
    LoaderConfig,
    SEN12MSCRDataConfig,
    ShuffleConfig,
    Trainer,
    TrainerConfig,
    build_sen12mscr_dataloader,
)


class TinyCloudRemovalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 13, kernel_size=1),
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([sar.float(), cloudy.float()], dim=1))


class FloatMSELoss(nn.Module):
    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs, target.float())


data_config = SEN12MSCRDataConfig(
    split_strategy="seeded_scene",
    seed=7,
    loader=LoaderConfig(batch_size=4),
    shuffle=ShuffleConfig(buffer_size=16, reshard_num_shards=16),
)

train_loader = build_sen12mscr_dataloader("train", data_config)
val_loader = build_sen12mscr_dataloader("val", data_config)
test_loader = build_sen12mscr_dataloader("test", data_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCloudRemovalNet().to(device)

# 먼저 model을 옮기고, 그 다음 그 model 기준으로 optimizer를 만듭니다.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = FloatMSELoss()
metrics = {
    "mae": lambda outputs, target: F.l1_loss(outputs, target.float()),
}

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    metrics=metrics,
    config=TrainerConfig(
        max_epochs=2,
        train_max_batches=10,
        val_max_batches=2,
        test_max_batches=2,
        checkpoint_dir="artifacts/checkpoints",
    ),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

for history in trainer.step():
    print(history)

print(trainer.test())
```

번들 예제 실행:

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 8 --val-max-batches 2
```

## 데이터 계약

decode된 sample과 collate된 batch는 모두 같은 표준 스키마를 사용합니다.

```python
{
    "inputs": ...,
    "target": ...,
    "metadata": ...,
}
```

SEN12MS-CR에서는 구체적으로 다음과 같습니다.

- `inputs["sar"]`: SAR tensor
- `inputs["cloudy"]`: cloudy optical tensor
- `target`: cloud-free optical target tensor
- `metadata`: `season`, `scene`, `patch`, `source_shard` 같은 scene 및 decode metadata

즉 trainer 바깥의 학습 코드는 `outputs`와 `target` 기준으로 생각하면 되고, 모델은 `inputs`를 받습니다. `inputs`가 mapping이면 `Trainer`는 `model(**inputs)`를 호출합니다.

## Trainer 계약

`Trainer`는 다음을 받습니다.

- `model`
- `optimizer`
- `criterion`
- `metrics`
- `config`
- `train_loader`
- optional `val_loader`
- optional `test_loader`

실행 계약은 단순합니다.

- `criterion`은 `criterion(outputs, target)`으로 호출됩니다
- `metrics`는 `name -> callable(outputs, target)` mapping입니다
- 학습은 `for history in trainer.step(): ...`를 사용합니다
- 평가는 `trainer.test()`를 사용합니다

공개 `Trainer` API에는 `step_fn`, `DataModule`, scheduler가 없습니다.

## 데이터 빌더

```python
from cr_train import (
    SEN12MSCRDataConfig,
    build_sen12mscr_dataset,
    build_sen12mscr_dataloader,
)
```

공개 빌더는 다음입니다.

- `SEN12MSCRDataConfig`
- `build_sen12mscr_dataset(stage, config, transform=None)`
- `build_sen12mscr_dataloader(stage, config, transform=None, collate_fn=None)`

`stage`는 `train`, `val`, `test`입니다.

sample 단위 변환이 필요하면 `transform`을 쓰고, 표준 sample 스키마 위에 custom batching을 얹고 싶으면 `collate_fn`을 사용하면 됩니다.

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

scene split 안전성을 유지하면서도 학습 시 mixing을 확보하기 위한 고정 규칙입니다.

### 5. Reproducibility

기본 loader 설정은 재현성을 우선합니다.

- `num_workers=0`
- `in_order=True`
- `persistent_workers=False`

같은 seed, split strategy, batch size, loader topology면 같은 batch가 나옵니다. worker 수나 prefetch를 늘리면 처리량 위주의 모드로 보는 것이 맞습니다.

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
    LoaderConfig,
    SEN12MSCRDataConfig,
    ShuffleConfig,
    SplitRatios,
    Trainer,
    TrainerConfig,
    build_sen12mscr_dataloader,
    build_sen12mscr_dataset,
)
```

주요 설정 포인트:

- `SEN12MSCRDataConfig.split_strategy`: `official` 또는 `seeded_scene`
- `SEN12MSCRDataConfig.seed`: split과 shuffle에 공통으로 쓰는 seed
- `SEN12MSCRDataConfig.loader`: `LoaderConfig(...)`
- `SEN12MSCRDataConfig.shuffle`: `ShuffleConfig(...)`
- `ShuffleConfig.buffer_size`: streaming shuffle buffer 크기
- `ShuffleConfig.reshard_num_shards`: shuffle 전 reshard 목표값
- `LoaderConfig.batch_size`: batch size
- `TrainerConfig.max_epochs`: `trainer.step()`이 생성할 전체 epoch 수
- `TrainerConfig.train_max_batches`, `val_max_batches`, `test_max_batches`: partial epoch 제한
- `TrainerConfig.checkpoint_dir`: optional checkpoint 디렉터리

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

개발 및 테스트 의존성 설치:

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
