# magicnet-cloud

Репозиторий подготовлен для запуска в Google Colab без создания отдельного виртуального окружения.

Что изменено:

- зависимости для Colab ставятся скриптом `script/install_colab_requirements.sh`
- путь к каталогу с датасетами задаётся через верхнеуровневый параметр `data_dir`
- датасетные конфиги собирают свои пути через `${data_dir}/...`
- добавлен ноутбук для тестового запуска обучения и просмотра TensorBoard

Быстрый старт:

```bash
bash script/install_colab_requirements.sh
python examples/classification/main.py --cfg cfgs/scanobjectnn/dgcnn.yaml data_dir=/content/data epochs=1 wandb.use_wandb=False
```

Для сегментации через `script/main_segmentation.sh` в Colab дополнительные зависимости теперь подхватываются автоматически.
Например, `PointNet++`-запуск можно стартовать сразу так:

```bash
bash script/main_segmentation.sh cfgs/k3d_xyz/pointnet++/pointnet++.yaml data_dir=/content/data epochs=1 wandb.use_wandb=False
```

Ноутбук для Colab:

- [notebooks/google_colab_train_smoke.ipynb](/home/researcher/dev/magicnet-cloud/notebooks/google_colab_train_smoke.ipynb)
