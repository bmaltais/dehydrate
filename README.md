# Dehydrate and rehydrate model

A way to "dehydrate" a dreambooth model down to less than 1GB and "rehydrate" it back to it's original self (same hash). This technically allow to save half the storage space at the cost of requiring rehydration. This could be dynamically be implemented in txt2img or img2img given the right hydration code support.

This is how it goes.

1. Use the ckpt_subtract.py script to subtract the original model from the DB model, leaving behind only the difference between the two.
2. Compress the difference model to 1GB or less
3. To rehydrate the model simply reverse the process. Add the diff back on top of the original sd15 model with ckpt_add.py.

Dehydrate:

`python .\ckpt_subtract.py D:\models\sks_man-1e-6-3000-sd15.ckpt D:\models\v1-5-pruned.ckpt --output diff.ckpt`

Hydrate:

`python .\ckpt_add.py .\diff.ckpt D:\models\v1-5-pruned.ckpt --output D:\models\sks_man-1e-6-3000-sd15.ckpt`