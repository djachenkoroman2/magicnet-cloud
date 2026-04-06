# magicnet-cloud

Репозиторий подготовлен для запуска из Google Colab или другого ноутбука без отдельного виртуального окружения. Ниже оставлена только актуальная для ноутбука информация: установка зависимостей, запуск `script/main_classification.sh` и `script/main_segmentation.sh`, а также настройка путей к датасетам и логам на Google Drive.

## Подготовка в Google Colab

Сначала клонируйте репозиторий в Colab и перейдите в его каталог:

```bash
git clone https://github.com/djachenkoroman2/magicnet-cloud.git
cd /content/magicnet-cloud
```

Если репозиторий уже был клонирован в текущем runtime, достаточно перейти в каталог проекта:

```bash
cd /content/magicnet-cloud
```

Смонтируйте Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Рекомендуемые каталоги:

- датасеты: `/content/drive/MyDrive/data`
- логи и чекпоинты: `/content/drive/MyDrive/logs`

Создать их можно так:

```bash
mkdir -p /content/drive/MyDrive/data
mkdir -p /content/drive/MyDrive/logs
```

Установите зависимости текущего Colab-runtime:

```bash
bash script/install_colab_requirements.sh
```

Примечание для Colab: если вы запускаете команды через отдельные `!`-ячейки, переменные окружения между ними не сохраняются. Поэтому либо используйте абсолютные пути прямо в команде, либо запускайте команды через `%%bash`.

## Как устроены пути к данным и логам

В базовом конфиге проекта заданы:

- `data_dir`: общий корень с датасетами
- `log_root`: общий корень для логов
- `root_dir: ${log_root}`: фактический корень, куда складываются run-директории

Это значит:

- если вы передаёте `data_dir=/content/drive/MyDrive/data`, конкретный конфиг сам достроит путь к нужному датасету
- если вы передаёте `log_root=/content/drive/MyDrive/logs`, логи и чекпоинты будут сохраняться на Google Drive

Итоговая структура логов создаётся автоматически в виде:

```text
<log_root>/<task_name>/<run_name>/
```

Например:

- классификация `cfgs/scanobjectnn/...` пишет в `/content/drive/MyDrive/logs/scanobjectnn/<run_name>/`
- сегментация `cfgs/k3d_xyz/...` пишет в `/content/drive/MyDrive/logs/k3d_xyz/<run_name>/`

Если путь на Google Drive содержит пробелы или кириллицу, передавайте override целиком в кавычках:

```bash
"data_dir=/content/drive/MyDrive/Мои данные/data"
"log_root=/content/drive/MyDrive/Мои данные/logs"
```

## Где должны лежать датасеты

### Classification: ScanObjectNN

Для конфигов из `cfgs/scanobjectnn/*.yaml` проект ожидает данные по пути:

```text
<data_dir>/ScanObjectNN/h5_files/main_split/
```

Минимально нужны файлы:

```text
training_objectdataset_augmentedrot_scale75.h5
test_objectdataset_augmentedrot_scale75.h5
```

Если вы передали:

```text
data_dir=/content/drive/MyDrive/data
```

то итоговый путь должен быть таким:

```text
/content/drive/MyDrive/data/ScanObjectNN/h5_files/main_split/
```

### Classification: Birds

Для конфигов из `cfgs/birds/*.yaml` проект теперь тоже ожидает общий корень `data`:

```text
data_dir=/content/drive/MyDrive/data
```

а сам датасет должен лежать внутри:

```text
/content/drive/MyDrive/data/birds/
```

Структура папки `birds`:

```text
<data_dir>/birds/
  <class_1>/
    sample_001.txt
    sample_002.txt
  <class_2>/
    sample_001.txt
  ...
  splits/
    train.txt
    val.txt
    test.txt
```

Что важно:

- каждая папка верхнего уровня, кроме `splits/`, считается отдельным классом
- каждый `.txt` файл должен содержать как минимум три колонки: `x y z`
- папка `splits/` необязательна; если её нет, train/val/test будут собраны автоматически по `split_ratios` из конфига

Для `birds` в `main_classification.sh` теперь достаточно передать общий корень:

```bash
--data /content/drive/MyDrive/data
```

### Segmentation: K3DXYZ

Для конфигов из `cfgs/k3d_xyz/*.yaml` проект ожидает:

```text
<data_dir>/k3d_xyz/
  raw/
  splits/
  processed/
```

Обязательно должны существовать:

- `raw/` с исходными сценами
- `splits/train.txt`, `splits/val.txt`, `splits/test.txt`

Каталог `processed/` может быть создан автоматически во время работы.

Если вы передали:

```text
data_dir=/content/drive/MyDrive/data
```

то итоговый путь будет:

```text
/content/drive/MyDrive/data/k3d_xyz/
```

Если датасет лежит не по стандартной схеме, можно переопределить точный путь напрямую через `dataset.common.data_root=/полный/путь`.

## `script/main_classification.sh`

### Назначение

Скрипт запускает `examples/classification/main.py`, сам находит конфиг и умеет отдельно принимать путь к датасету, корень логов и чекпоинт для продолжения обучения.

### Синтаксис

```bash
bash script/main_classification.sh <config_path> [--data <dataset_path>] [--log <log_root>] [--resume <checkpoint_path>] [extra args...]
```

### Что означает каждый аргумент

- `<config_path>`: путь к yaml-конфигу, например `cfgs/scanobjectnn/dgcnn.yaml`
- `--data`: корень датасетов; скрипт сам подставит его в нужный ключ конфига
- `--log`: корень для логов и чекпоинтов
- `--resume`: путь к checkpoint; скрипт автоматически добавит `mode=resume`
- `[extra args...]`: любые дополнительные override параметров конфига, например `epochs=1`, `mode=test`, `runtime.device=cpu`

### Как работает `--data`

`main_classification.sh` специально сделан так, чтобы не заставлять вас помнить внутренний ключ конфига:

- если конфиг использует `dataset.common.data_dir`, будет переопределён именно он
- если конфиг использует `dataset.common.data_root`, будет переопределён он
- иначе будет переопределён верхнеуровневый `data_dir`

Для `birds` это тоже работает: если передать общий корень `data`, датасет автоматически найдёт папку `data/birds`.

Для `cfgs/scanobjectnn/*.yaml` это удобно, потому что достаточно передать только корень:

```bash
bash script/main_classification.sh \
  cfgs/scanobjectnn/dgcnn.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  epochs=1
```

В этом случае классификация будет искать ScanObjectNN здесь:

```text
/content/drive/MyDrive/data/ScanObjectNN/h5_files/main_split/
```

Логи и чекпоинты уйдут сюда:

```text
/content/drive/MyDrive/logs/scanobjectnn/<run_name>/
```

Для `cfgs/birds/*.yaml` теперь тоже можно передавать общий каталог `/content/drive/MyDrive/data`:

```bash
bash script/main_classification.sh \
  cfgs/birds/pointnet.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  epochs=1
```

В этом случае классификация будет читать данные из:

```text
/content/drive/MyDrive/data/birds/
```

Логи и чекпоинты уйдут сюда:

```text
/content/drive/MyDrive/logs/birds/<run_name>/
```

### Типовые сценарии

Обучение на `ScanObjectNN`:

```bash
bash script/main_classification.sh \
  cfgs/scanobjectnn/dgcnn.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  epochs=50
```

Тест `ScanObjectNN` по готовому чекпоинту:

```bash
bash script/main_classification.sh \
  cfgs/scanobjectnn/dgcnn.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  mode=test \
  pretrained_path=/content/drive/MyDrive/logs/scanobjectnn/<run_name>/checkpoint/<ckpt_best>.pth
```

Продолжение обучения `ScanObjectNN` с последнего checkpoint:

```bash
bash script/main_classification.sh \
  cfgs/scanobjectnn/dgcnn.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  --resume /content/drive/MyDrive/logs/scanobjectnn/<run_name>/checkpoint/<ckpt_latest>.pth
```

Обучение на `birds`:

```bash
bash script/main_classification.sh \
  cfgs/birds/pointnet.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  epochs=50
```

Тест `birds` по готовому чекпоинту:

```bash
bash script/main_classification.sh \
  cfgs/birds/pointnet.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  mode=test \
  pretrained_path=/content/drive/MyDrive/logs/birds/<run_name>/checkpoint/<ckpt_best>.pth
```

Продолжение обучения `birds` с последнего checkpoint:

```bash
bash script/main_classification.sh \
  cfgs/birds/pointnet.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  --resume /content/drive/MyDrive/logs/birds/<run_name>/checkpoint/<ckpt_latest>.pth
```

Запуск `birds` на CPU:

```bash
bash script/main_classification.sh \
  cfgs/birds/pointnet.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  runtime.device=cpu \
  epochs=1
```

Запуск `ScanObjectNN` на CPU:

```bash
bash script/main_classification.sh \
  cfgs/scanobjectnn/dgcnn.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  runtime.device=cpu \
  epochs=1
```

## `script/main_segmentation.sh`

### Назначение

Скрипт запускает `examples/segmentation/main.py` и поддерживает те же основные флаги для путей, что и `script/main_classification.sh`: `--data`, `--log`, `--resume`.

### Синтаксис

```bash
bash script/main_segmentation.sh <config_path> [--data <dataset_path>] [--log <log_root>] [--resume <checkpoint_path>] [extra args...]
```

### Что означает каждый аргумент

- `<config_path>`: путь к yaml-конфигу, например `cfgs/k3d_xyz/pointnet++/pointnet++.yaml`
- `--data`: общий корень датасетов; если в конфиге есть верхнеуровневый `data_dir`, будет переопределён именно он
- `--log`: корень логов и чекпоинтов; если в конфиге есть `log_root`, будет переопределён он
- `--resume`: путь к checkpoint для продолжения обучения; скрипт автоматически добавит `mode=resume`
- `[extra args...]`: любые остальные override параметров конфига, например `epochs=1`, `mode=test`, `pretrained_path=...`, `runtime.device=cpu`

### Как передавать дополнительные override

Все дополнительные параметры по-прежнему можно передавать в формате:

```bash
ключ=значение
```

Примеры:

- `epochs=1`
- `mode=val`
- `pretrained_path=/content/drive/MyDrive/.../checkpoint/model_ckpt_best.pth`
- `runtime.device=cpu`
- `dataset.test.split=val`

### Важная особенность для Colab

Если скрипт понимает, что запущен в Google Colab, он автоматически вызывает `script/install_colab_requirements.sh` и при необходимости пытается поставить дополнительные зависимости для сегментационных моделей.

Это особенно полезно для:

- `PointNet++`
- `PointNeXt`
- `PointTransformer`

Если автоподготовка не нужна, её можно отключить:

```bash
SKIP_COLAB_REQUIREMENTS=1 bash script/main_segmentation.sh ...
```

### Базовый запуск для K3DXYZ

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  epochs=1
```

В этом случае сегментация будет читать данные из:

```text
/content/drive/MyDrive/data/k3d_xyz/
```

А run-директория появится здесь:

```text
/content/drive/MyDrive/logs/k3d_xyz/<run_name>/
```

### Типовые сценарии

Обучение:

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  epochs=100
```

Продолжение обучения:

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  --resume /content/drive/MyDrive/logs/k3d_xyz/<run_name>/checkpoint/<ckpt_latest>.pth
```

Валидация лучшего checkpoint:

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  mode=val \
  pretrained_path=/content/drive/MyDrive/logs/k3d_xyz/<run_name>/checkpoint/<ckpt_best>.pth
```

Тест:

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  mode=test \
  pretrained_path=/content/drive/MyDrive/logs/k3d_xyz/<run_name>/checkpoint/<ckpt_best>.pth
```

Тест по списку `val.txt` вместо `test.txt`:

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  mode=test \
  dataset.test.split=val \
  pretrained_path=/content/drive/MyDrive/logs/k3d_xyz/<run_name>/checkpoint/<ckpt_best>.pth
```

Запуск на CPU:

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --data /content/drive/MyDrive/data \
  --log /content/drive/MyDrive/logs \
  runtime.device=cpu \
  epochs=1
```

### Когда использовать `data_dir`, а когда `dataset.common.data_root`

Через `--data` скрипт в первую очередь переопределяет верхнеуровневый `data_dir`, поэтому это основной и рекомендуемый способ. Для K3DXYZ:

```bash
--data /content/drive/MyDrive/data
```

Тогда конфиг сам превратит его в:

```text
/content/drive/MyDrive/data/k3d_xyz
```

Используйте `dataset.common.data_root`, если датасет лежит в нестандартной папке:

```bash
bash script/main_segmentation.sh \
  cfgs/k3d_xyz/pointnet++/pointnet++.yaml \
  --log /content/drive/MyDrive/logs \
  dataset.common.data_root=/content/drive/MyDrive/custom_datasets/my_k3d_xyz \
  epochs=1
```

## TensorBoard для логов на Google Drive

Если логи сохраняются на Google Drive, TensorBoard можно открыть прямо по корню логов:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/logs
```

Или по конкретной задаче:

```python
%tensorboard --logdir /content/drive/MyDrive/logs/k3d_xyz
```

## Коротко: что использовать в ноутбуке

- для классификации удобнее всего: `bash script/main_classification.sh ... --data <путь_к_датасетам> --log <путь_к_логам>`
- для сегментации удобнее всего: `bash script/main_segmentation.sh ... --data <путь_к_датасетам> --log <путь_к_логам>`
- если датасет лежит не по стандартной структуре конфига, переопределяйте точный путь через `dataset.common.data_root=...`
- если чекпоинты и логи должны сохраняться на Google Drive, передавайте `log_root=/content/drive/MyDrive/logs`
