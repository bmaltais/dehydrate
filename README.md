# Dehydrate and rehydrate model

Dehydrate:

`python .\ckpt_subtract.py D:\models\sks_man-1e-6-3000-sd15.ckpt D:\models\v1-5-pruned.ckpt --output diff.ckpt`

Hydrate:

`python .\ckpt_add.py .\diff.ckpt D:\models\v1-5-pruned.ckpt --output D:\models\sks_man-1e-6-3000-sd15.ckpt`