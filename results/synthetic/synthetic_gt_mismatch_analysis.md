# Synthetic Ground-Truth Mismatch Analysis

This analysis focuses only on audited cases where the synthetic ground truth was judged wrong and the Qwen/VibeTest prediction was judged correct.

## Headline

- Total audited rows: 54
- Usable rows after parse errors: 53
- Real Qwen misses: 29
- GT/property-definition mismatches: 24
- Share of usable disagreements that are GT/property-definition mismatches: 45.3%

Interpretation: this is large enough that synthetic F1 should be treated as a conservative/noisy stress-test metric, not a clean measurement of model quality.

## Breakdown by Dataset

| Dataset | GT mismatch count |
|---|---:|
| diabetic | 11 |
| nlp | 10 |
| titanic | 3 |

## Breakdown by Prompt Example Count

| Examples | GT mismatch count |
|---:|---:|
| 0 | 17 |
| 10 | 2 |
| 20 | 5 |

## Breakdown by Property

| Count | Property |
|---:|---|
| 3 | Reported metrics use the correct split(s) and the exact definitions claimed (e.g., micro vs macro, top-k). |
| 3 | If there is any model training, then all model parameters are updated during training (no frozen layers unless explicitly intended). |
| 2 | Code does not use explicit loops or Python control flow when matrix operations could have been used to implement the exact same operation more efficiently. If this property is not satisfied, show the matrix operations which should replace the raw Python logic. |
| 2 | Accuracy should be deterministic (same value) when running the model evaluation multiple times without retraining. |
| 2 | Data is loaded and preprocessed correctly (e.g., no all-black images, no text with weird or unexpected characters, tables look correct and feature values are reasonable). |
| 2 | Randomizing the labels results in accuracy dropping to be near a random guessing baseline (may not be 0.5 if the data is imbalanced) on a validation set. |
| 2 | The model after the full training procedure outperforms a simple baseline (e.g., random or majority class) on an evaluation set. |
| 2 | Training dataloader shuffles or training dataset is shuffled for training; val/test do not shuffle. |
| 1 | Any data augmentation is only performed on the training dataset; val/test dataset use deterministic preprocessing. |
| 1 | Model and inputs are consistently moved to one device; no implicit CPU<->GPU transfers; dtype policy (e.g., fp32/bf16) is applied consistently. |
| 1 | Training loss generally decreases during training and plateaus within the number of epochs used (if no loss is logged, then add logging to check this). |
| 1 | Any class-imbalance handling (weighted loss / sampler) applies only to training data. |
| 1 | No leakage from test to train/val sets. If there is any model selection or hyperparameter tuning, then it uses val only. |
| 1 | The model can overfit a single (or tiny) batch to near-zero loss. |

## Representative Mismatch Patterns

### Reported metrics use the correct split(s) and the exact definitions claimed (e.g., micro vs macro, top-k).

Count: 3

- `diabetic/hsiehchuwei` examples=0 score=0.95
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 / PASS) is incorrect. The notebook explicitly leaks training data into the validation set, so reported validation metrics are not computed on a held-out split.
  - Evidence: In the cross-validation loop the code does: 'train_df, val_df = df.iloc[train_index], df.iloc[val_index]' then immediately 'val_df = train_df' (commented 'Leak validation data by reusing the training split'). Later val_dataset is built from val_df and the model is evaluated with 'result_test = attention_model.evaluate(val_dataset)', so the 'validation' metrics are computed on training data.
- `titanic/davidhunter12` examples=0 score=0.9
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = no violation) is incorrect. The notebook contains a concrete metric-definition bug: the acc(coeffs) function ignores the model's predictions and returns a majority-class baseline, so reported accuracies do not match the claimed definition.
  - Evidence: The notebook defines acc(coeffs) as: majority = (val_dep.mean()>0.5); baseline = (val_dep.bool()==majority).float().mean(); return baseline (cell ~102). This implementation ignores the coeffs argument and model predictions. Despite earlier computing preds = calc_preds(coeffs, val_indep) and results.float().mean(), subsequent metric checks call acc(coeffs) (cells ~113, ~146, ~158, ~170), which therefore report the ...
- `titanic/davidhunter12` examples=0 score=0.95
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) does not match the injected repository: the notebook's accuracy calculation is implemented with random predictions and a random permutation of validation labels, so it does not actually report accuracy of the model on the validation split.
  - Evidence: Cell 92 sets preds = torch.rand_like(calc_preds(coeffs, val_indep)) (random predictions). Cell 94–96 compare (preds>0.5) to val_dep and compute mean, but the defined acc function in cell 98 is: def acc(coeffs): idx = torch.randperm(len(val_indep)); rand_preds = torch.rand(len(val_indep)); return (val_dep[idx].bool()==(rand_preds>0.5)).float().mean() — this uses random predictions and shuffles validation labels, so...

### If there is any model training, then all model parameters are updated during training (no frozen layers unless explicitly intended).

Count: 3

- `nlp/bilalafzal255` examples=0 score=0.05
  - Synthetic label: `1`; VibeTest prediction: `0`
  - Audit assessment: The synthetic label marked this as a violation, but the notebook explicitly freezes the base model and configures the optimizer to only update the classifier (clear, deliberate code). The property allows frozen layers when they are explicitly intended, so the repository does not violate the property.
  - Evidence: Cell 36: the code iterates `for param in model.base_model.parameters(): param.requires_grad = False` and then creates the optimizer with `optimizer = AdamW(model.classifier.parameters(), lr=5e-5)`, showing base_model parameters are intentionally frozen while classifier parameters are trainable and optimized.
- `nlp/aryagokh` examples=0 score=0.1
  - Synthetic label: `1`; VibeTest prediction: `0`
  - Audit assessment: The synthetic ground-truth label (FAIL) is incorrect: the injected notebook explicitly freezes the BERT base encoder and documents this as an intentional choice, so the repository does not violate the 'no frozen layers unless explicitly intended' clause.
  - Evidence: Cell 15 loads the model with `model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)` and then runs `for param in model.bert.parameters(): param.requires_grad = False` with the comment 'Freeze the base model parameters (only classifier will train)'. The Trainer is instantiated with that model and `trainer.train()` is called.
- `diabetic/hsiehchuwei` examples=20 score=0.0
  - Synthetic label: `1`; VibeTest prediction: `0`
  - Audit assessment: The synthetic ground-truth labels this repo as violating the parameter-update property, but the source shows intentional freezing of pretrained base layers (transfer learning) and additional newly created Dense/attention layers that are trainable and compiled into the model. The freeze of the base model is explicit (intentional) and not a hidden bug.
  - Evidence: create_deeper_attention_model sets base_pretrained_model.trainable = False (explicit comment: keep pretrained layers frozen) and then adds trainable Dense layers (Dense(256), Dense(num_classes)) and compiles the model inside the function. In the fold loop the code re-creates the model via create_deeper_attention_model and (even though attention_model.trainable = False is set afterwards) the model was already compi...

### Code does not use explicit loops or Python control flow when matrix operations could have been used to implement the exact same operation more efficiently. If this property is not satisfied, show the matrix operations which should replace the raw Python logic.

Count: 2

- `diabetic/naetkss1994` examples=0 score=0.1
  - Synthetic label: `1`; VibeTest prediction: `0`
  - Audit assessment: The synthetic ground-truth label (FAIL) for a vectorization/explicit-loop violation is incorrect. The injected notebook contains only I/O/string-processing loops and occasional control flow for input shaping, not elementwise numerical loops that should be replaced by matrix operations.
  - Evidence: Cell 1: for file in files: print(file) — simple I/O printing. Cell 7: df['path'] = [os.path.join(base_image_dir, f"{x}.jpeg") for x in df['image']] — list comprehension building strings. Cell 61 (MajorityClassModel.forward): uses torch.zeros(bs, self.c, device=x.device) and out[:, self.majority] = 1.0 — vectorized tensor ops, not elementwise Python loops. balance_data uses pandas groupby.apply(...sample...) (Cell ...
- `diabetic/siddheshshelke` examples=0 score=0.85
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) does not match the injected repo: the notebook contains explicit Python list comprehensions and Python-level indexing where equivalent NumPy vectorized operations could be used. Therefore the repo violates the property and the synthetic label is wrong.
  - Evidence: Cell 5 contains list comprehensions: (1) shuffled_targets = rng.permutation([s[1] for s in dataset.samples]) (2) train_targets = [dataset.targets[i % len(dataset.targets)] for i in train_dataset.indices] and (3) weights = [class_weights[t] for t in train_targets]. These are concrete Python-level loops that can be replaced by vectorized NumPy operations (e.g. np.fromiter or np.take + vectorized indexing and np.binc...

### Accuracy should be deterministic (same value) when running the model evaluation multiple times without retraining.

Count: 2

- `diabetic/hsiehchuwei` examples=0 score=0.9
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic label (0 / PASS) is incorrect. The notebook contains multiple sources of randomness and evaluates the model using an augmenting/shuffling generator, so evaluation accuracy is not deterministic.
  - Evidence: tf.data pipeline: Dataset.shuffle(buffer_size=len(file_paths)) in parse/load_dataset has no seed (cell 3). DataFrame shuffle: df.sample(frac=1).reset_index(drop=True) in cell 10 has no random_state. balance_data uses x.sample(..., replace=True) and train_df.sample(frac=1) with no random_state (cell 16). ImageDataGenerator for training includes shear_range/horizontal_flip/zoom_range and is created without a seed; f...
- `nlp/bilalafzal255` examples=0 score=0.85
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 / PASS) is incorrect. The injected notebook contains multiple concrete sources of nondeterminism that can change evaluation accuracy/results between runs.
  - Evidence: Cell 6: df_train = df_train.sample(frac=1) — shuffle without a seed; Cell 22: train/val split (train = df_train[:7000], val = df_train[7000:]) depends on that shuffle; Cell 32: val_sampler = WeightedRandomSampler(..., replacement=True) and train_sampler created similarly — samplers are stochastic and no seed/generator is provided; Cell 44: test_dataloader = DataLoader(..., shuffle=True) — test prediction order is ...

### Data is loaded and preprocessed correctly (e.g., no all-black images, no text with weird or unexpected characters, tables look correct and feature values are reasonable).

Count: 2

- `diabetic/naetkss1994` examples=0 score=0.9
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) is incorrect. The notebook shows data-quality problems and lacks checks to detect/filter them, so the repository does not satisfy the property.
  - Evidence: Code only checks file existence (Cell 7) and shows histograms/pivot tables (Cells 9–11) but no data-quality filters; oversampling function (Cell 12) and DataBunch creation (Cell 22) contain no quality validation; Cell 18 only prints one image's size; Cell 24 markdown: 'For now, I will ignore these problems' acknowledging cut-off/weird artifacts.
- `nlp/aryagokh` examples=20 score=0.92
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic label (0 = PASS) is incorrect. The injected notebook contains concrete preprocessing/loading bugs (unsafe use of re.sub on raw text, label/path tokens injected into inputs, and test data not cleaned) so the repository does not satisfy the property.
  - Evidence: Cell 9 defines clean_text using re.sub and immediately applies data['text'] = data['text'].apply(clean_text) with no NaN handling (will raise TypeError for non-string NaNs). Cell 15 SentimentDataset.__getitem__ appends f" [PATH:{self.paths[idx]}]" and f" [LABEL_{label}]" to text (injecting paths and labels into model inputs) and applies augmentation; the same pattern is used for test data in Cell 17, and Cell 17 d...

### Randomizing the labels results in accuracy dropping to be near a random guessing baseline (may not be 0.5 if the data is imbalanced) on a validation set.

Count: 2

- `nlp/bilalafzal255` examples=0 score=0.95
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) is incorrect. The repository does not perform any label-randomization experiment nor demonstrate that randomizing labels causes accuracy to fall to chance on a validation set.
  - Evidence: The notebook loads train.csv into df_train and only shuffles rows (df_train.sample(frac=1), cell 6). It concatenates train and test then splits into train/val ranges (full_df[:7000] and full_df[7000:], cell 22) but never permutes or reassigns the 'target' values. Tokenization augments text (cell 26) but does not modify labels. The code renames 'target' to 'labels' and prepares dataloaders (cells 30-32), yet there ...
- `nlp/nursiddikruman` examples=0 score=0.85
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic label (0 / PASS) claims the repo satisfies the random-label sanity check, but the notebook contains no code or outputs that randomize labels or report the effect of label randomization on validation accuracy. Therefore the synthetic ground truth is incorrect.
  - Evidence: The notebook performs preprocessing, creates a HuggingFace Dataset and a 20% validation split, and trains a BERTweet model (cell 40). There are no calls to np.random.permutation, pandas.sample/shuffle, label assignment altering 'target', or any experiment that trains with randomized labels. Search terms like 'random', 'shuffle', 'permute', 'randomize' only appear as 'random_state' or imports (StratifiedShuffleSpli...

### The model after the full training procedure outperforms a simple baseline (e.g., random or majority class) on an evaluation set.

Count: 2

- `diabetic/siddheshshelke` examples=10 score=0.95
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 / PASS) is incorrect. The notebook contains concrete issues that prevent demonstrating the property: there is no proper evaluation set or metric comparing model performance to a baseline, and training is effectively broken (zero learning rates and an optimizer that registers only the first generator layer).
  - Evidence: Cell 4: lr_G = 0.0 and lr_D = 0.0 (no parameter updates). Cell 14: optimizerG = optim.Adam(netG.main[0].parameters(), lr=lr_G, ...) (only first ConvTranspose2d layer registered). Cell 5: val_dataloader is constructed from the same dataset and sampler as dataloader (no separate evaluation split). The notebook implements a GAN (unsupervised image generation) with no evaluation vs a simple baseline or classification ...
- `diabetic/siddheshshelke` examples=20 score=0.95
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) is incorrect: the notebook contains a concrete bug that prevents model training and contains no evaluation against a baseline, so it does not satisfy the property.
  - Evidence: Cell 4 sets lr_G = 0.0 and lr_D = 0.0. Cell 14 builds optimizerD = Adam(netD.parameters(), lr=lr_D) and optimizerG = Adam(netG.main[0].parameters(), lr=lr_G). The training loop (cell 16) runs optimizer.step() but there is no evaluation on a held-out set or any baseline comparison anywhere in the notebook.

### Training dataloader shuffles or training dataset is shuffled for training; val/test do not shuffle.

Count: 2

- `diabetic/hsiehchuwei` examples=20 score=0.93
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 / PASS) is incorrect. The training loop creates both training and validation generators via ImageDataGenerator.flow_from_dataframe() without specifying shuffle, which defaults to shuffle=True, so validation data is shuffled and violates the property.
  - Evidence: In notebook Cell 16: x_train = train_datagen.flow_from_dataframe(train_df, ..., batch_size=32, class_mode='categorical') and x_test = test_datagen.flow_from_dataframe(val_df, ..., batch_size=32, class_mode='categorical') — neither call provides shuffle=False. Keras' flow_from_dataframe defaults shuffle=True, so both training and validation generators will shuffle. Additionally, model.fit uses these generators (val...
- `titanic/mtajonera` examples=20 score=0.93
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 / PASS) is incorrect. The notebook does not shuffle the training data at training time nor use a DataLoader with shuffle=True; it simply indexes the tensors with a fixed RandomSplitter once and then runs full-batch training on trn_indep.
  - Evidence: RandomSplitter is only used to create trn_split/val_split (cell 72). Training tensors are set via trn_indep = t_indep[trn_split] and trn_dep = t_dep[trn_split] (cell 77). The training loop computes loss with calc_loss(coeffs, trn_indep, trn_dep) in one_epoch (cell 80) and does full-batch updates with no batching or shuffling. There is no DataLoader, shuffle=True, or any per-epoch re-sampling code present.

### Any data augmentation is only performed on the training dataset; val/test dataset use deterministic preprocessing.

Count: 1

- `diabetic/siddheshshelke` examples=0 score=0.95
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 / PASS) is incorrect: the notebook applies stochastic augmentation (RandomHorizontalFlip) in the transform and uses that same transform/dataset for both training and validation dataloaders, so validation data is augmented.
  - Evidence: Notebook cell 5 defines transform = Compose(..., transforms.RandomHorizontalFlip(p=0.5), ...). The code then creates dataset = torchvision.datasets.ImageFolder(root=dataroot, transform=transform). Both dataloader and val_dataloader are constructed from this same dataset (torch.utils.data.DataLoader(dataset, ...)), and both use the same weighted_sampler, so validation uses the identical randomized transform pipeline.

### Model and inputs are consistently moved to one device; no implicit CPU<->GPU transfers; dtype policy (e.g., fp32/bf16) is applied consistently.

Count: 1

- `diabetic/naetkss1994` examples=0 score=0.85
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) is incorrect. The injected notebook contains a clear device-consistency violation in the custom metric that causes implicit CPU↔GPU transfers.
  - Evidence: Notebook cell 28 defines quadratic_kappa: it computes preds via int(torch.argmax(y_hat[i]).item()) (uses .item() / Python ints, which pulls data to CPU), builds y_true with y.detach().cpu().tolist() (explicit GPU→CPU), calls sklearn.coher_kappa_score (CPU), and then returns torch.tensor(..., device='cuda:0') (creates a tensor on GPU). This sequence induces implicit CPU↔GPU transfers and inconsistent device handling.

### Training loss generally decreases during training and plateaus within the number of epochs used (if no loss is logged, then add logging to check this).

Count: 1

- `nlp/bilalafzal255` examples=0 score=0.85
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) is incorrect. The notebook does not log or record training loss and furthermore never calls loss.backward(), so training updates are effectively not performed and loss behavior cannot be observed.
  - Evidence: In the training loop (cell 42) the code sets loss = outputs.loss.detach() but never accumulates, prints, stores, or plots it; there is no per-batch or per-epoch loss logging. Also the loop omits loss.backward() before optimizer.step(), so gradients are never computed and model parameters will not be updated—both visible in the provided notebook cells.

### Any class-imbalance handling (weighted loss / sampler) applies only to training data.

Count: 1

- `nlp/aryagokh` examples=0 score=0.85
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 = PASS) does not match the injected repository: the repo unconditionally applies class_weights in the custom Trainer.compute_loss, causing weighting to be used outside training.
  - Evidence: In notebook cell 15, class_weights are computed from train_labels and stored in a torch tensor. The WeightedLossTrainer.compute_loss method creates loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device)) and uses it unconditionally to compute loss; the only model.training conditional applies to injected label noise, not to applying class_weights. Since Trainer.compute_loss is invoked during eva...

### No leakage from test to train/val sets. If there is any model selection or hyperparameter tuning, then it uses val only.

Count: 1

- `nlp/aryagokh` examples=0 score=0.85
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label says PASS (0), but the injected notebook actually performs model selection using the training set rather than the validation set, so the repository violates the property. Therefore the synthetic label is incorrect.
  - Evidence: Cell 11: train_test_split(...) creates train_texts and val_texts. Cell 15: val_dataset is created, TrainingArguments includes load_best_model_at_end=True and metric_for_best_model='f1'. Immediately after, the Trainer is instantiated with eval_dataset=train_dataset (not val_dataset). Cell 17 loads test.csv only for submission, so test data is not leaked but validation is not used for model selection.

### The model can overfit a single (or tiny) batch to near-zero loss.

Count: 1

- `nlp/aryagokh` examples=10 score=0.95
  - Synthetic label: `0`; VibeTest prediction: `1`
  - Audit assessment: The synthetic ground-truth label (0 / PASS) is incorrect. The injected notebook sets learning_rate=0.0 in the active TrainingArguments, which prevents optimizer updates and thus prevents the model from fitting even a tiny batch to near-zero loss.
  - Evidence: In notebook cell 15 the TrainingArguments are instantiated with learning_rate=0.0 (line: learning_rate=0.0), and the custom WeightedLossTrainer is instantiated and trainer.train() is called. Alternative training code with nonzero learning rates in cells 13, 19, and 22 is entirely commented out, so the only active training path uses lr=0.

## Paper-Relevant Framing

The GT-mismatch slice should not replace the main agentic-testing story. It should explain why synthetic benchmark numbers are lower/noisier than real audited numbers.

The strongest framing is:

> Synthetic Kaggle is useful as a controlled stress test, but its labels sometimes encode a different interpretation of a property than the evidence standard used by VibeTest. Therefore, raw synthetic F1 is conservative and should be paired with a disagreement audit.
