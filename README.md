# MALARIA-CNN

Binary classification of **parasitized vs. uninfected** blood-cell images with a custom
convolutional neural network, using the public **NLM/NIH malaria cell-image dataset**.

The defining feature of this project is the **group-aware data partitioning**: every
image is tied to its source group (`group_id`, e.g. `C101P62ThinF`), and no group is ever
split across the training, validation, and test subsets — so the model cannot win by
memorising slide/patient-specific artefacts.

> **Scope:** image-level binary classification research prototype.
> Not a medical device, not clinically validated, no external validation performed.

---

## Headline results (independent test set, n = 6,457)

| Metric | Value |
|---|---|
| Accuracy | **94.89%** |
| Precision | **97.45%** |
| Sensitivity (recall) | **93.56%** |
| Specificity | **96.69%** |
| F1-score | **95.46%** |
| ROC-AUC | **98.71%** |
| False positive rate | 3.31% |
| False negative rate | 6.44% |
| Test loss | 0.1512 |

Confusion matrix (threshold 0.5, parasitized = positive class):

| | Predicted Uninfected | Predicted Parasitized |
|---|---|---|
| **Actual Uninfected** | 2,656 (TN) | 91 (FP) |
| **Actual Parasitized** | 239 (FN) | 3,471 (TP) |

Per-class report:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Uninfected | 0.9174 | 0.9669 | 0.9415 | 2,747 |
| Parasitized | 0.9745 | 0.9356 | 0.9546 | 3,710 |

All values above are read directly from `result/results_baseline_cnn.csv` and the saved
notebook output — nothing is rounded or re-derived.

---

## Dataset

- **Source:** U.S. National Library of Medicine / NIH —
  <https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html>
  (also mirrored as TensorFlow Datasets `malaria`).
- **Images:** 27,558 RGB PNG cell images, perfectly balanced —
  13,779 `Parasitized` + 13,779 `Uninfected`.
- **Image geometry (measured over all 27,558 files):**

  | | Min | Max | Mean |
  |---|---|---|---|
  | Width | 46 | 394 | 132.49 |
  | Height | 40 | 385 | 132.98 |

- **Integrity check:** all 27,558 files opened and verified with PIL — **0 corrupted**.

### Source-group metadata

Two mapping files (`ground-truth metadata/`) list, for each source group, the cell
filenames belonging to it. They are read with `header=None` because the **first source
group identifier is real data, not a column label** — reading with an inferred header
silently drops that row (27,433 records / 200 groups instead of 27,558 / 201).

After correct parsing:

| Property | Value |
|---|---|
| Records | 27,558 |
| Unique filenames | 27,558 |
| Unique `group_id` values | 201 |
| Duplicate filename records | 0 |
| Groups containing **both** classes | 150 |

The mixed-label groups are exactly why stratification alone is not enough: the split
must be group-aware *and* label-aware at the same time.

---

## Group-aware partitioning

Implemented with `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
in two stages:

1. First fold → **test** (set aside, never touched during development)
2. From the remaining data, first fold → **validation**

| Subset | Images | Uninfected | Parasitized | Share | Groups |
|---|---|---|---|---|---|
| Train | 16,496 | 8,766 (53.14%) | 7,730 (46.86%) | 59.9% | 128 |
| Validation | 4,605 | 2,266 (49.21%) | 2,339 (50.79%) | 16.7% | 33 |
| Test | 6,457 | 2,747 (42.54%) | 3,710 (57.46%) | 23.4% | 40 |
| **Total** | **27,558** | **13,779** | **13,779** | 100% | **201** |

Leakage checks (re-verified from the committed CSVs):

- `train ∩ val` groups: **0**
- `train ∩ test` groups: **0**
- `val ∩ test` groups: **0**
- duplicated filenames across all three splits: **0**

> Note: the test set is *not* class-balanced (57.5% parasitized) — `StratifiedGroupKFold`
> preserves class proportions per fold, not perfect equality.

---

## Preprocessing & augmentation

**Pipeline** (applied in `tf.data`):

1. `tf.io.read_file` → `tf.image.decode_png(channels=3)`
2. `tf.image.convert_image_dtype(..., tf.float32)` → pixels in `[0, 1]`
3. `tf.image.resize_with_pad(128, 128)` → fixed input, **aspect ratio preserved**, zero padding
4. shuffle (`buffer_size = len(train)`, `seed = 42`) → batch 32 → `prefetch(AUTOTUNE)`

**Augmentation** lives *inside* the model as a `Sequential` block, so it runs only while
`training=True` and is automatically bypassed on validation/test:

| Layer | Setting |
|---|---|
| `RandomFlip` | `horizontal_and_vertical` |
| `RandomRotation` | `factor = 0.08` (±28.8°) |
| `RandomZoom` | `height/width_factor = 0.10` |
| `RandomTranslation` | `height/width_factor = 0.05` |
| `RandomContrast` | `factor = 0.10` |

---

## Model

Custom CNN, input `128 × 128 × 3`:

| Layer / component | Configuration |
|---|---|
| Input | `128 × 128 × 3` (RGB) |
| Augmentation | see table above |
| Conv block ×4 | `Conv2D(32/64/128/256, 3×3, padding="same")` → `BatchNormalization` → `ReLU` |
| Pooling | `MaxPooling2D(2×2)` after stages 1–3 (stage 4 has none) |
| Head | `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout(0.4)` → `Dense(1, sigmoid)` |

| Parameters | Count |
|---|---|
| Total | 423,361 (1.61 MB) |
| Trainable | 422,401 |
| Non-trainable | 960 (batch-norm scale/bias) |

Global average pooling is used instead of a flatten+dense stack to keep the parameter
count small.

---

## Training configuration

| Setting | Value |
|---|---|
| Optimizer | Adam, `lr = 1e-3` |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping | `monitor=val_loss`, `patience=5`, `restore_best_weights=True` |
| Reduce LR on plateau | `factor=0.3`, `patience=2`, `min_lr=1e-6` |
| Model checkpoint | `baseline_cnn_best.keras`, `monitor=val_loss`, `save_best_only` |
| Tracked metrics | accuracy, precision, recall, AUC |

**Actual run:** stopped at **epoch 16**; best `val_loss = 0.1228` at **epoch 11**;
best `val_accuracy = 95.81%` at **epoch 9**; learning rate decayed
`1e-3 → 3e-4 → 9e-5 → 2.7e-5 → 8.1e-6 → 2.43e-6`.

The test set was evaluated **once**, after training and model selection were complete.

---

## Repository structure

```
MALARIA-CNN/
├── README.md
├── .gitignore
├── code/
│   └── Malaria.ipynb            # full pipeline: metadata parsing → split → train → evaluate
├── data-split/
│   ├── train.csv                # 16,496 rows
│   ├── validation.csv           # 4,605 rows
│   └── test.csv                 # 6,457 rows
├── ground-truth metadata/
│   ├── patientid_cellmapping_parasitized.csv
│   └── patientid_cellmapping_uninfected.csv
├── model/
│   └── baseline_cnn_best.keras  # best checkpoint (val_loss, epoch 11)
├── result/
│   └── results_baseline_cnn.csv # final test metrics
└── dataset/
    └── cell_images.zip          # NOT committed (337 MB > GitHub 100 MB limit)
```

Split CSV columns: `filename, group_id, label, filepath, exists`.

> `filepath` holds the **absolute Colab path** from the original run
> (`/content/malaria_project/data/...`). It will not resolve on your machine —
> rebuild paths from `filename` + `label` (both are in the CSV).

---

## Reproducing

**Environment** (as used): Google Colab, Python 3, TensorFlow/Keras,
pandas, NumPy, scikit-learn (`StratifiedGroupKFold`), Matplotlib, Pillow.

1. Download `cell_images.zip` from the NLM/NIH link above and place it in `dataset/`.
2. Upload `ground-truth metadata/*.csv` alongside the notebook.
3. Open `code/Malaria.ipynb` in Colab and run all cells in order.
4. The notebook writes `train/validation/test.csv`, saves `baseline_cnn_best.keras`,
   and emits `results_baseline_cnn.csv`.

Notebook cell order matters: the metadata is parsed twice — the first pass (with an
inferred header) is superseded by the `header=None` pass that yields the full
27,558 records.

---

## Acknowledgements

Dataset courtesy of the **U.S. National Library of Medicine** and the **National
Institutes of Health**.
