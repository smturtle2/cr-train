# cr-train

[English](./README.md) | 한국어

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

결정론적 스트리밍 실험을 위한 모델 비의존적 SEN12MS-CR 학습 유틸리티입니다.

## 주요 기능

- **결정론적 스트리밍** -- scene 단위 분할과 epoch 간 재현 가능한 배치 순서
- **모델 비의존적** -- 어떤 PyTorch `nn.Module`이든 사용 가능; `forward(sar, cloudy)`만 구현하면 됨
- **자동 I/O 튜닝** -- worker 수, prefetch, parquet readahead 자동 설정
- **체크포인트 관리** -- RNG 상태 포함 전체 저장으로 정확한 재개 가능
- **내장 진행 표시** -- `tqdm.rich` stage bar에 실시간 loss/metric 업데이트

## 프로젝트 구조

```
cr-train/
├── src/cr_train/           # 핵심 패키지
│   ├── __init__.py         #   공개 API export
│   ├── trainer.py          #   Trainer, TrainerConfig, MAE
│   ├── data.py             #   데이터셋 로딩 및 전처리
│   └── runtime.py          #   parquet I/O 튜닝
├── examples/
│   ├── minimal_train.py    # 기준 학습 스크립트
│   └── colab_quickstart.ipynb
├── tests/
├── scripts/
└── pyproject.toml
```

## 빠른 시작

```bash
git clone https://github.com/smturtle2/cr-train.git
cd cr-train
uv sync
```

Hugging Face rate limit 완화를 위해 토큰 설정을 권장합니다:

```bash
export HF_TOKEN=your_token
```

기준 예제 실행:

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 10 --val-max-batches 2
```

Colab 노트북도 있습니다: [`examples/colab_quickstart.ipynb`](./examples/colab_quickstart.ipynb)

## 사용법

이 저장소는 클론해서 로컬 학습 모듈처럼 사용하는 형태입니다.

1. 클론 및 설치: `git clone ... && uv sync`
2. 저장소 루트에서 스크립트를 실행하면 `cr_train` 패키지를 바로 import할 수 있습니다.
3. `src.cr_train`이 아니라 `cr_train`에서 import합니다:

```python
from cr_train import Trainer, TrainerConfig, MAE, build_sen12mscr_loaders

train_loader, val_loader, test_loader = build_sen12mscr_loaders(batch_size=4)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=nn.MSELoss(),
    metrics=[MAE()],
    config=TrainerConfig(max_epochs=5),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

for history in trainer.step():
    print(history["train"], history["val"])

test_metrics = trainer.test()
```

자체 스크립트를 둘 때의 일반적인 레이아웃:

- `examples/` -- 재현 가능한 기준 예제
- `scripts/` -- 일회성 실험 스크립트
- 저장소 루트 -- `uv run python your_script.py`로 빠른 실험

기준 구현은 [`examples/minimal_train.py`](./examples/minimal_train.py)입니다.

## 공개 API

### `build_sen12mscr_loaders`

```python
build_sen12mscr_loaders(
    batch_size,
    *,
    seed=0,
    split="official",              # "official" | "seeded_scene"
    shuffle_buffer_size=16,
    num_workers=None,              # None = 자동 조정
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=None,
    io_profile="smooth",           # "smooth" | "conservative"
) -> tuple[DataLoader, DataLoader, DataLoader]
```

반환값은 `(train_loader, val_loader, test_loader)`입니다.

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `num_workers` | `None` | CPU 코어 수 기반 자동 조정 |
| `io_profile` | `"smooth"` | parquet readahead/threading 완만 적용; 동기식 I/O는 `"conservative"` 사용 |
| `persistent_workers` | `None` | worker가 켜진 경우 재사용 |

### `Trainer`

```python
Trainer(
    model,                         # nn.Module
    optimizer,                     # torch.optim.Optimizer
    criterion,                     # loss 함수
    metrics,                       # 예: [MAE()]
    config,                        # TrainerConfig
    train_loader,
    val_loader=None,
    test_loader=None,
)
```

**메서드:**

| 메서드 | 설명 |
|--------|------|
| `trainer.step()` | epoch별 history dict를 yield하는 iterator |
| `trainer.test()` | test loader 평가, metrics dict 반환 |
| `trainer.save_checkpoint(path)` | 모델 + optimizer + RNG 상태 저장 |
| `trainer.load_checkpoint(path)` | 체크포인트에서 복원 |

**진행 표시:**

- 기본적으로 `tqdm.rich` 사용; `show_progress=False`로 비활성화.
- 각 epoch는 먼저 헤더를 출력하고, 그 아래에 stage bar가 렌더링됨.
- 첫 배치 전 `loading first batch...`를 표시하고, 이후 배치마다 `loss`와 metric을 갱신.
- `max_batches`는 불필요한 데이터를 미리 가져오지 않고 깔끔하게 중단.

**내장 metric:** `MAE` (Mean Absolute Error).

## 배치 계약

각 sample과 collate된 batch는 다음 스키마를 따릅니다:

```python
{
    "inputs": {
        "sar":    Tensor,    # [B, 2, H, W]   SAR 후방산란
        "cloudy": Tensor,    # [B, 13, H, W]  구름 낀 광학 영상
    },
    "target":     Tensor,    # [B, 13, H, W]  구름 없는 광학 영상
    "metadata": {
        "season": list[str],
        "scene":  list[str],
        "patch":  list[str],
        ...
    },
}
```

`Trainer`는 `inputs`를 모델에 전달하고, `criterion(output, target)`과 각 metric을 `(output, target)`에 적용합니다.

## 데이터 기본값

| 설정 | 값 |
|------|-----|
| Split | `official` (175 scenes: 155 train / 10 val / 10 test) |
| Optical 전처리 | `clip(0, 10000) / 10000.0` -> float32 |
| SAR 전처리 | `clip(-25, 0)`, 이후 `(x + 25) / 25` -> [0, 1] |
| Train 파이프라인 | `reshard() -> shuffle(seed, buffer_size)` -> batch |
| Scene 구성 | epoch 간 고정; train 순서만 결정론적으로 변경 |
| Dataset revision | 재현성을 위해 고정 |

## 재현성

- Scene 단위 분할로 train, validation, test 간 scene 누수를 방지합니다.
- Official split은 저자 supplementary metadata 기반으로 번들됩니다.
- 같은 seed와 loader 설정이면 배치 순서를 완전히 재현할 수 있습니다.

## Official Split 출처

| 자료 | 링크 |
|------|------|
| 공식 페이지 | <https://patricktum.github.io/cloud_removal/sen12mscr/> |
| Supplementary 폴더 | <https://u.pcloud.link/publink/show?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV> |
| 직접 받는 `splits.csv` | <https://api.pcloud.com/getpubtextfile?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV&fileid=57823192235> |
| 번들된 manifest | [`official_scene_splits.csv`](./src/cr_train/resources/official_scene_splits.csv) |
| 갱신 스크립트 | [`refresh_official_scene_splits.py`](./scripts/refresh_official_scene_splits.py) |
