$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe -u -m training.train_classifier `
  --data_dir data\plantnet_cls `
  --labelmaps training\labelmaps.json `
  --epochs 20 `
  --batch_size 32 `
  --img_size 224
