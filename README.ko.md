# cr-train

[English](./README.md) | 한국어

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

deterministic streaming 실험을 위한 모델-불가지론적 SEN12MS-CR 학습 유틸리티입니다.

## 클론 및 설치

```bash
git clone https://github.com/smturtle2/cr-train.git
cd cr-train
uv sync
```

Colab 스타일 온보딩 노트북:

- [`examples/colab_quickstart.ipynb`](/home/smturtle2/projects/cr-train/examples/colab_quickstart.ipynb)

노트북도 환경 구성은 `uv`를 쓰고, 실제 notebook runtime은 Colab 또는 Jupyter가 담당합니다.

Hugging Face rate limit을 줄이려면 토큰 설정을 권장합니다.

```bash
export HF_TOKEN=your_token
```

## 권장 사용 흐름

이 저장소는 클론해서 로컬 학습 모듈처럼 쓰는 형태를 전제로 합니다. 보통은 다음 순서로 사용합니다.

1. 저장소를 클론하고 `uv sync`로 환경을 맞춥니다.
2. 스크립트 실행은 저장소 루트에서 유지합니다. 그러면 프로젝트 환경 안에서 로컬 `cr_train` 패키지를 바로 import할 수 있습니다.
3. 먼저 동작하는 기준점이 필요하면 예제 엔트리포인트부터 실행합니다.

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 10 --val-max-batches 2
```

4. 직접 학습 스크립트를 만들 때는 `src.cr_train`이 아니라 `cr_train`에서 import합니다.
   보통 `build_sen12mscr_loaders`, `Trainer`, `TrainerConfig`, 필요하면 `MAE` 같은 metric을 가져오면 됩니다.
5. `(train_loader, val_loader, test_loader)`를 만들고, 모델/optimizer/loss를 준비한 뒤 `Trainer`에 넘깁니다.
6. 학습은 `for history in trainer.step(): ...`, 평가는 `trainer.test()`로 붙입니다.

자체 스크립트를 이 저장소 안에 둘 때는 보통 이런 식으로 나눕니다.

- `examples/`: 재현 가능한 기준 예제
- `scripts/`: 일회성 실험 스크립트
- 저장소 루트 또는 하위 디렉터리 스크립트를 `uv run python ...`으로 실행

기준 예제는 [`minimal_train.py`](/home/smturtle2/projects/cr-train/examples/minimal_train.py)입니다.

## 공개 API

```python
build_sen12mscr_loaders(
    batch_size,
    *,
    seed=0,
    split="official",
    shuffle_buffer_size=16,
    num_workers=None,
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=None,
    io_profile="smooth",
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
- `train`, `val`, `test` 진행 바는 기본적으로 `tqdm.rich`를 사용하며, `show_progress=False`로 끌 수 있습니다.
- 각 epoch는 먼저 헤더를 출력하고, 그 아래에 stage 진행 바가 표시되며 완료 후에도 남아 있습니다.
- 각 stage는 첫 배치 전 `loading first batch...`를 먼저 보여주고, 이후에는 배치마다 running `loss`와 metrics를 갱신합니다. 길이를 아는 loader는 마지막 배치 update가 끝난 뒤에만 다음 stage로 넘어갑니다.
- `max_batches` 제한은 더 이상 사용하지 않을 다음 배치를 미리 가져오지 않아서, stage 경계에서 불필요하게 멈추지 않습니다.
- 기본 로더 경로는 더 부드러운 streaming을 우선합니다. `num_workers=None`은 worker 수를 자동 조정하고, `persistent_workers=None`은 worker가 켜진 경우 재사용하며, `io_profile="smooth"`는 parquet readahead/threading을 완만하게 켭니다. 이전의 동기식 동작으로 돌아가려면 `num_workers=0`과 `io_profile="conservative"`를 사용하면 됩니다.

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
- scene 구성 자체는 epoch마다 고정되고, train 순서만 현재 epoch에 따라 deterministic하게 바뀝니다.
- 기본 loader 설정은 auto worker와 `smooth` parquet I/O profile로 epoch 중간 load pause를 줄이는 쪽으로 맞춰져 있습니다.
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
