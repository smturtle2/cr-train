# cr-train

[English](./README.md) | 한국어

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

deterministic streaming 실험을 위한 모델-불가지론적 SEN12MS-CR 학습 유틸리티입니다.

## 설치

```bash
uv sync
```

Colab 스타일 온보딩 노트북:

- [`examples/colab_quickstart.ipynb`](/home/smturtle2/projects/cr-train/examples/colab_quickstart.ipynb)

노트북도 환경 구성은 `uv`를 쓰고, 실제 notebook runtime은 Colab 또는 Jupyter가 담당합니다.

Hugging Face rate limit을 줄이려면 토큰 설정을 권장합니다.

```bash
export HF_TOKEN=your_token
```

## 빠른 시작

```python
import torch
from torch import nn

from cr_train import MAE, Trainer, TrainerConfig, build_sen12mscr_loaders


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
        return self.net(torch.cat([sar, cloudy], dim=1))


train_loader, val_loader, test_loader = build_sen12mscr_loaders(
    batch_size=4,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCloudRemovalNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    metrics=[MAE()],
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

## 공개 API

```python
build_sen12mscr_loaders(
    batch_size,
    *,
    seed=0,
    split="official",
    shuffle_buffer_size=16,
    num_workers=0,
    pin_memory=False,
)
```

반환값은 `(train_loader, val_loader, test_loader)`입니다.

```python
Trainer(
    model,
    optimizer,
    criterion,
    metrics,
    config,
    train_loader,
    val_loader=None,
    test_loader=None,
)
```

- `metrics`는 metric object 시퀀스입니다. 예: `metrics=[MAE()]`
- 내장 metric object는 `MAE`입니다.
- 학습은 `for history in trainer.step(): ...`로 실행합니다.
- 평가는 `trainer.test()`로 실행합니다.

## 배치 계약

decode된 sample과 collate된 batch는 모두 같은 스키마를 사용합니다.

```python
{
    "inputs": {
        "sar": ...,
        "cloudy": ...,
    },
    "target": ...,
    "metadata": ...,
}
```

`Trainer`는 `inputs`를 모델에 전달하고, `criterion(outputs, target)`과 각 metric object를 `(outputs, target)`에 적용합니다.

## 데이터 기본값

- 기본 split은 `official`입니다.
- optical 전처리 기본값은 `cloudy`, `target` 모두 `clip(0, 10000)` 후 `/ 10000.0`으로 `float32` 변환입니다.
- SAR 전처리 기본값은 `clip(-25, 0)` 후 `(x + 25) / 25`로 `[0, 1]` 구간에 매핑합니다.
- train 파이프라인은 batching 전에 항상 `reshard() -> shuffle(seed, buffer_size)` 순서를 유지합니다.
- 재현성을 위해 dataset revision은 pin됩니다.

## 재현성

- scene-level split으로 train, validation, test 간 scene 누수를 막습니다.
- official split은 저자 supplementary metadata를 기반으로 번들됩니다.
- 같은 seed와 loader 설정이면 batch 순서를 재현할 수 있습니다.

## Official Split 출처

- 공식 페이지: <https://patricktum.github.io/cloud_removal/sen12mscr/>
- supplementary 폴더: <https://u.pcloud.link/publink/show?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV>
- 직접 받는 `splits.csv`: <https://api.pcloud.com/getpubtextfile?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV&fileid=57823192235>
- 번들된 정규화 manifest: [`official_scene_splits.csv`](/home/smturtle2/projects/cr-train/src/cr_train/resources/official_scene_splits.csv)
- 갱신 스크립트: [`refresh_official_scene_splits.py`](/home/smturtle2/projects/cr-train/scripts/refresh_official_scene_splits.py)
