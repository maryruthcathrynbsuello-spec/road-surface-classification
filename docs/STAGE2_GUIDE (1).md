# Stage 2 Guide: Baseline CNN + Transfer Learning

Notebooks: `notebooks/02_baseline_cnn.ipynb` and `notebooks/03_transfer_learning.ipynb`
Textbook step: **Data Mining** (the data from Stage 1 is turned into a model that finds the pattern).

For every step below: **What** it is, **How** it works, **Why** we need it.

---

## Big picture

| Part | Goal |
|---|---|
| Notebook 02 | Train a small CNN from scratch. This is the **baseline**: the score that anything fancier must beat. |
| Notebook 03 | Train a CNN that starts from knowledge learned on a huge image collection (**transfer learning**). Expect it to be better, especially on a small dataset. |
| Comparison | Put both results side by side. This is your Results section. |

Only **training** and **validation** images are used here. The **test** set is not touched until Stage 3.

| Set | Used for |
|---|---|
| Train | The model learns from these |
| Validation | We check progress during training and pick the best settings |
| Test | One final, honest score in Stage 3 |

---

## Step 0: Check what you need first
- `processed_dataset.zip` and `preprocessing_config.json` in `road_project/outputs/` (made by notebook 01)
- Colab with **Runtime -> Change runtime type -> T4 GPU**
- Everyone uses the same shared files and `SEED = 42`

**Why:** If teammates use different data or seeds, their scores can't be compared.

## Step 1: Setup and GPU check (notebook cell "0. Setup")
**What:** Imports libraries, connects Google Drive, sets the random seed, and prints whether a GPU is found.
**How:** `set_random_seed(42)` fixes the random numbers used for weight initialization and shuffling.
**Why:** A CNN does millions of multiplications per image. A GPU does many at once, so training takes minutes instead of hours. A fixed seed makes your results repeatable.

## Step 2: Settings and `RUN_NAME`
**What:** One cell with the numbers you can change: epochs, batch size, learning rate, dropout.
**How:** Each experiment gets its own name (`baseline_cnn_v1`, `baseline_cnn_v2`...). Files saved to Drive use that name.
**Why:** You will try several versions. Without unique names, you overwrite old results and can't prove in your paper which setting gave which score.

| Setting | Meaning |
|---|---|
| Epoch | One full pass through all training images |
| Batch size | How many images the model looks at before updating itself (32) |
| Learning rate | How big each update step is. Too big = unstable, too small = very slow |
| Dropout | Randomly switches off a share of neurons during training so the model can't just memorize |

## Step 3: Load the data
**What:** Unzips the dataset onto Colab's fast local disk, reads the class names and class weights from the config file, and builds the train and validation pipelines.
**How:**
- Images are read in **batches** of 32, so memory never fills up.
- `prefetch` loads the next batch while the GPU trains on the current one.
- `class_names` come from the config file, so the order of classes is the same everywhere (notebooks, app).
- **Class weights** were computed in Stage 1 from the training set. A rare class gets a larger weight.

**Why:**
- Class order must never differ between training and the app, or "pothole" would be shown as "open manhole".
- Without class weights, a model can score high by mostly guessing the biggest class (usually "normal road") and ignoring rare ones.
- The test set is deliberately **not loaded** here, so nobody can peek at it.

## Step 4: Build the baseline CNN (notebook 02)
**What:** A small network built from repeated blocks, then a classifier.

| Layer | How it works | Why it is there |
|---|---|---|
| **Augmentation** (flip, rotate, zoom, brightness, contrast) | Randomly changes each training image a little. Switched off automatically for validation and testing. | Shows the model many variations, so it learns the road damage and not the exact photo. Reduces overfitting. |
| **Rescaling 1/255** | Turns pixel values 0-255 into 0-1 | Neural networks train more smoothly on small numbers |
| **Conv2D** | Slides small filters (3x3) over the image to detect patterns: edges first, then textures, then shapes like cracks or holes | This is what makes it a CNN. Filters are learned, not hand-made. |
| **BatchNormalization** | Keeps the numbers in each layer on a stable scale | Faster, more stable training |
| **ReLU** | Sets negative values to zero | Lets the network learn non-straight-line patterns |
| **MaxPooling** | Keeps the strongest value in every 2x2 area, halving the image size | Fewer numbers to compute, and small shifts in position matter less |
| **GlobalAveragePooling** | Averages each feature map into one number | Much fewer parameters than flattening, so less overfitting |
| **Dropout** | Randomly drops neurons while training | Prevents memorizing |
| **Dense + softmax (6 outputs)** | Turns the features into 6 probabilities that add up to 1 | Gives "crack 7%, pothole 85%, ..." |

The filters grow 32 -> 64 -> 128 -> 256 across four blocks: early layers see simple edges, later layers see complex shapes.
Size: about 0.4 million parameters (the numbers the model learns).

**Why a baseline at all:** It tells you how hard the problem is. If a tiny CNN already scores 90%, transfer learning's gain is small. If it scores 50%, the data is hard or something is wrong. It is also the reference your better model must beat.

## Step 5: Compile (choose the loss and optimizer)
**What:** Tells the model how to measure mistakes and how to improve.
**How:**
- **Loss = sparse categorical crossentropy.** Measures how far the predicted probabilities are from the right class. Confident wrong answers are punished hard.
- **Optimizer = Adam.** Adjusts all weights a little after each batch in the direction that lowers the loss.
- **Metric = accuracy**, shown for you to read.

**Why:** Training is just "make the loss smaller" over and over. "Sparse" means labels are plain numbers (0-5), which is how our folder-based labels are stored.

## Step 6: Train with callbacks
**What:** `model.fit(...)` repeats: show a batch -> predict -> measure loss -> adjust weights. After each epoch it checks the validation set.
**Callbacks (automatic helpers):**

| Callback | What it does | Why |
|---|---|---|
| **EarlyStopping** | Stops when validation loss hasn't improved for 8 epochs | Stops before the model starts memorizing |
| **ModelCheckpoint** | Saves the best model so far to Drive | If Colab disconnects, you keep the best version |
| **ReduceLROnPlateau** | Halves the learning rate when progress stalls | Big steps early, careful steps later |
| **class_weight** | Mistakes on rare classes cost more | Fair treatment of imbalanced classes |

After training, the notebook reloads the **best** saved model (not just the last epoch).

## Step 7: Read the learning curves
**What:** Two charts: accuracy and loss for train vs validation, over epochs.
**Why:** This is how you see if training went well.

| What you see | Meaning | What to try |
|---|---|---|
| Train and validation both improve and stay close | Healthy | Nothing |
| Train much better than validation | **Overfitting** (memorizing) | More dropout, stronger augmentation, fewer filters |
| Both low and flat | **Underfitting** | More epochs, bigger model, higher learning rate |
| Validation jumps up and down | Unstable | Lower learning rate, bigger batch size |

Put the curves in your paper and explain what they show.

## Step 8: Validation results
**What:** Predicts the whole validation set and prints a report and a confusion matrix.
**How:**
- **Precision:** when the model says "pothole", how often is it right?
- **Recall:** of all real potholes, how many did it find?
- **F1:** one number combining precision and recall.
- **Macro F1:** the average F1 over classes, so every class counts equally.
- **Confusion matrix:** rows = true class, columns = predicted class. Numbers off the diagonal are mistakes.

**Why:** Overall accuracy can hide problems. A model that never detects open manholes can still show 90% accuracy if manholes are rare. Per-class F1 and the confusion matrix show it. Look especially at crack vs damaged asphalt and pothole vs water-filled pothole.

## Step 9: Save and log the run
**What:** The notebook saves the model (`models/<run>.keras`), the history and metrics (`outputs/history_<run>.json`, `metrics_<run>.json`) and the figures.
**Then you:** add a row to `docs/EXPERIMENT_LOG.md` and commit the figures to `docs/figures/`.
**Why:** Evidence for your paper, and the model file is what the app loads later. Never commit the `.keras` file to GitHub.

---

# Notebook 03: Transfer learning

## Step 10: The idea
**What:** Use a network (MobileNetV2 or EfficientNetB0) already trained on ImageNet, a collection of over a million photos in 1000 categories.
**How:** Its early layers already know edges, textures, and shapes. Those are useful for roads too. We keep that knowledge and only teach it our 6 classes.
**Why:**
- Needs far fewer road images to reach a good score
- Trains faster
- Usually beats a model built from scratch

| | MobileNetV2 | EfficientNetB0 |
|---|---|---|
| Size | About 2.3 million parameters | Larger, about 4 million |
| Speed | Faster, good for the app | A bit slower |
| Start with | **Yes** | Try as the second experiment |

Each expects its pixels prepared differently. The notebook handles this: MobileNetV2 gets values scaled to -1..1, EfficientNetB0 gets raw 0-255 because it scales inside.

## Step 11: Phase 1, train only the new head
**What:** The pretrained part (the **base**) is **frozen**. Only a new small classifier on top is trained.
**How:** `base.trainable = False`, then train for up to 15 epochs at learning rate 1e-3. The base is called with `training=False` so its BatchNorm layers keep their learned statistics.
**Why:** The new head starts with random weights. If we trained everything at once, its large random errors would flow back and **damage** the good pretrained features. So we let the head settle first.

## Step 12: Phase 2, fine-tune the top layers
**What:** Unfreeze the top 30 layers of the base and train them gently together with the head.
**How:**
- Learning rate drops to **1e-5**, 100x smaller. Pretrained weights only need small nudges.
- BatchNorm layers stay frozen (standard practice).
- You must **recompile** after changing which layers are trainable.
- Training continues counting epochs from where phase 1 stopped, so the curve is one continuous line. A dashed line marks where fine-tuning starts.

**Why:** The top layers hold the most task-specific features (ImageNet's "dog ear", "car wheel"). Adapting them to "crack" and "pothole" gives the extra accuracy. Unfreezing everything or using a big learning rate would erase the pretrained knowledge.

## Step 13: Compare all runs
**What:** The last cell reads every `metrics_*.json` and shows one table.
**Why:** This table is the core of your Results section: baseline vs MobileNetV2 vs EfficientNetB0 on validation accuracy, macro F1, and per-class F1.

---

# Improving results (the tuning loop)

1. Look at the curves and the confusion matrix.
2. Change **one** thing.
3. Rename the run (`_v2`) and train again.
4. Log it. Keep or discard.

| Problem | Try |
|---|---|
| Overfitting | More dropout, stronger augmentation, fewer fine-tuned layers |
| Underfitting | More epochs, more fine-tuned layers, bigger backbone |
| One class always weak | Check its images for label problems; more samples for it; check class weights |
| Crack vs damaged asphalt confusion | Review labels in Stage 0; consider higher resolution (e.g. 299) |
| Stuck at the same accuracy | Lower learning rate; check the data loaded correctly |

Stop when validation results stop improving across 2-3 experiments. Chasing tiny gains by tuning on validation too long will overfit your choices to the validation set.

## Common problems

| Problem | Likely cause and fix |
|---|---|
| "No GPU" warning | Runtime -> Change runtime type -> T4 GPU, rerun from the top |
| Out of memory | Lower `BATCH_SIZE` to 16 |
| Accuracy equals the biggest class's share | Model is guessing one class. Check class weights, learning rate, and that labels loaded correctly |
| Loss becomes `nan` | Learning rate too high, or corrupt images slipped through |
| Session disconnected | Rerun setup, then load the checkpoint from `models/` |
| `FileNotFoundError` for the zip or config | The Drive shortcut is missing, or notebook 01 has not been run by the owner |
| Different teammates get different results | Different data or seed. Everyone must use the shared `processed_dataset.zip` and `SEED = 42` |

---

# Who does what

| Person | Task |
|---|---|
| Model owner A | Run notebook 02, log it |
| Model owner B | Run notebook 03 with MobileNetV2, then EfficientNetB0 |
| Data owner | Check weak classes against the images; fix labels or data if needed (rerun notebook 01 once, then share the new files with everyone) |
| Docs owner | Write the Methodology section from this guide, add the curves and tables |

# What to write in your paper (Stage 2)
- Why CNN, and why two models
- The baseline architecture (table of layers) and the transfer-learning setup (backbone, two phases)
- Training settings: optimizer, learning rate, batch size, callbacks, class weights, augmentation
- Learning curves and what they show
- Validation accuracy, macro F1, per-class F1, and the comparison table
- Weak classes and what you tried

# Ready for Stage 3 when
- [ ] At least the baseline and one transfer model are trained and logged
- [ ] The best model is chosen using **validation** results only
- [ ] The model file is in `road_project/models/`
- [ ] The team agrees on which model is final
- [ ] Nobody has looked at the test set
