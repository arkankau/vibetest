# Direct Property Baseline Subset Comparison

Subset rows: 0, 1, 2, 3, 4

| method | n | coverage | macro F1 | acc all | acc covered | FAIL P | FAIL R | FAIL F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct property agent | 75 | 0.973 | 0.610 | 0.627 | 0.644 | 0.824 | 0.378 | 0.519 |
| VibeTest Qwen ex0 | 75 | 0.840 | 0.681 | 0.627 | 0.746 | 0.840 | 0.568 | 0.677 |
| VibeTest Qwen ex10 | 75 | 0.920 | 0.791 | 0.760 | 0.826 | 0.833 | 0.811 | 0.822 |
| VibeTest Qwen ex20 | 75 | 0.893 | 0.775 | 0.733 | 0.821 | 0.794 | 0.730 | 0.761 |

### Direct vs VibeTest Qwen ex0

Disagreements: 26 / 75

| row | property | GT | direct | VibeTest | direct reason |
|---:|---|---|---|---|---|
| 0 | kaggle_p0 | PASS | FAIL | PASS | The repository's main notebook uses the Titanic dataset without explicit data splitting into train, validation, and test sets. Model training and hyperparameter tuning (learning ra |
| 0 | kaggle_p11 | FAIL | FAIL | INCONCLUSIVE | The notebook trains directly on entire dataset tensors without using a dataloader or dataset shuffling. No evidence of training data shuffle or dataloader with shuffle parameter wa |
| 0 | kaggle_p12 | FAIL | PASS | FAIL | Evaluation uses a fixed trained model and a deterministic forward prediction function with no evidence of stochastic elements such as dropout or random sampling during evaluation.  |
| 0 | kaggle_p13 | FAIL | PASS | FAIL | The model and inputs run only on CPU as indicated by the environment metadata and absence of any explicit device or CUDA commands. No implicit CPU<->GPU transfers occur. Dtype is n |
| 0 | kaggle_p7 | PASS | FAIL | INCONCLUSIVE | The notebook includes a model training and accuracy evaluation but does not contain any experiment or code that randomizes labels and compares accuracy to a random guessing baselin |
| 0 | kaggle_p8 | FAIL | PASS | INCONCLUSIVE | The notebook trains a linear and deep learning model on the Titanic dataset, showing model accuracy over 80%. This accuracy exceeds typical majority class baseline (~60%) even thou |
| 1 | kaggle_p13 | PASS | PASS | FAIL | The notebook does not implement any device or dtype transfers indicating consistent implicit CPU usage and no transfers, thus no violation of device or dtype consistency. |
| 1 | kaggle_p4 | FAIL | PASS | INCONCLUSIVE | The repository notebook balances the dataset by downsampling classes before training, with no use of weighted loss or sampler for class imbalance handling seen anywhere. The balanc |
| 1 | kaggle_p6 | PASS | INCONCLUSIVE | PASS | The notebook showcases training and some accuracy reporting calls but lacks explicit information on the dataset splits used for metrics or the exact metric definitions claimed (e.g |
| 1 | kaggle_p9 | FAIL | PASS | INCONCLUSIVE | The repository contains a complete notebook that trains a deep neural net from scratch on a small balanced Titanic dataset, with manual gradient update steps and no explicit regula |
| 2 | kaggle_p0 | FAIL | PASS | FAIL | No test or validation split found, no hyperparameter tuning on test set seen, training done only on training dataset from Titanic comp input. |
| 2 | kaggle_p1 | PASS | PASS | INCONCLUSIVE | The notebook implements explicit model training with parameter updates applied to all model parameters (layers and constants). No parameters are left frozen. This satisfies the pro |
| 2 | kaggle_p12 | FAIL | PASS | FAIL | The notebook fixes model coefficients before evaluation, and the accuracy function uses these fixed coefficients to predict on static data, ensuring deterministic accuracy on repea |
| 2 | kaggle_p2 | FAIL | PASS | FAIL | The notebook demonstrates proper loading and preprocessing of the tabular Titanic dataset including missing value imputation, categorical variable encoding, and handling skewed num |
| 2 | kaggle_p3 | FAIL | PASS | FAIL | The notebook preprocessing consists of deterministic transformations only, with no evidence of augmentation on training or any differentiation vs validation/test preprocessing. Thu |
| 2 | kaggle_p9 | FAIL | PASS | FAIL | The notebook shows a from-scratch deep learning model trained with gradient descent on small batches without explicit regularization or early stopping to prevent overfitting. This  |
| 3 | kaggle_p0 | FAIL | PASS | FAIL | The notebook shows explicit random splitting of data into train/val sets, uses only training set for model training, only validation set for model selection/accuracy calculation, a |
| 3 | kaggle_p10 | PASS | PASS | INCONCLUSIVE | The repository contains training code in a notebook that explicitly calculates and prints the training loss every epoch during the training loop. The code comments indicate the los |
| 3 | kaggle_p11 | PASS | PASS | FAIL | Training and validation splits are created using fastai's RandomSplitter, which shuffles data indices internally for training, while validation indices are fixed and there is no fu |
| 3 | kaggle_p7 | PASS | FAIL | INCONCLUSIVE | The notebook contains code to measure validation accuracy on the correct labels but there is no code found that randomizes or shuffles the labels at any point to show the accuracy  |

### Direct vs VibeTest Qwen ex10

Disagreements: 28 / 75

| row | property | GT | direct | VibeTest | direct reason |
|---:|---|---|---|---|---|
| 0 | kaggle_p12 | FAIL | PASS | FAIL | Evaluation uses a fixed trained model and a deterministic forward prediction function with no evidence of stochastic elements such as dropout or random sampling during evaluation.  |
| 0 | kaggle_p13 | FAIL | PASS | FAIL | The model and inputs run only on CPU as indicated by the environment metadata and absence of any explicit device or CUDA commands. No implicit CPU<->GPU transfers occur. Dtype is n |
| 0 | kaggle_p6 | FAIL | PASS | FAIL | The repository uses a fixed index-based split for evaluation and calculates accuracy as the metric; this matches the implicit claim and correct definition of the metric used. No ev |
| 0 | kaggle_p8 | FAIL | PASS | INCONCLUSIVE | The notebook trains a linear and deep learning model on the Titanic dataset, showing model accuracy over 80%. This accuracy exceeds typical majority class baseline (~60%) even thou |
| 0 | kaggle_p9 | FAIL | PASS | FAIL | The notebook is an educational deep learning example for the Titanic dataset, showing typical model training steps without explicit constraints preventing single or tiny batch over |
| 1 | kaggle_p1 | FAIL | PASS | FAIL | The notebook explicitly shows a full manual implementation of training where all parameters are updated by gradient descent without freezing or skipping any parameters. |
| 1 | kaggle_p4 | FAIL | PASS | FAIL | The repository notebook balances the dataset by downsampling classes before training, with no use of weighted loss or sampler for class imbalance handling seen anywhere. The balanc |
| 1 | kaggle_p6 | PASS | INCONCLUSIVE | PASS | The notebook showcases training and some accuracy reporting calls but lacks explicit information on the dataset splits used for metrics or the exact metric definitions claimed (e.g |
| 1 | kaggle_p7 | FAIL | FAIL | PASS | The only file is a single-notebook copy of a known tutorial. It trains a model but does not contain any steps or code related to label randomization or testing accuracy drop upon l |
| 1 | kaggle_p8 | FAIL | PASS | FAIL | The notebook trains a deep learning model on the Titanic dataset from scratch and obtains over 80% accuracy on the evaluation set. This accuracy is higher than the known majority c |
| 1 | kaggle_p9 | FAIL | PASS | FAIL | The repository contains a complete notebook that trains a deep neural net from scratch on a small balanced Titanic dataset, with manual gradient update steps and no explicit regula |
| 2 | kaggle_p0 | FAIL | PASS | FAIL | No test or validation split found, no hyperparameter tuning on test set seen, training done only on training dataset from Titanic comp input. |
| 2 | kaggle_p12 | FAIL | PASS | FAIL | The notebook fixes model coefficients before evaluation, and the accuracy function uses these fixed coefficients to predict on static data, ensuring deterministic accuracy on repea |
| 2 | kaggle_p13 | FAIL | PASS | FAIL | The notebook uses PyTorch tensors to implement a model from scratch but contains no code for moving models or data to a specific device nor for consistent dtype policies. There is  |
| 2 | kaggle_p15 | FAIL | PASS | FAIL | The code uses efficient matrix operations for neural net calculations (e.g., torch matrix multiplication @). Non-matrix Python loops appear only for non-numeric operations like str |
| 2 | kaggle_p2 | FAIL | PASS | FAIL | The notebook demonstrates proper loading and preprocessing of the tabular Titanic dataset including missing value imputation, categorical variable encoding, and handling skewed num |
| 2 | kaggle_p3 | FAIL | PASS | FAIL | The notebook preprocessing consists of deterministic transformations only, with no evidence of augmentation on training or any differentiation vs validation/test preprocessing. Thu |
| 2 | kaggle_p5 | FAIL | PASS | FAIL | The notebook explicitly constructs path filenames as strings but only uses an explicitly derived numeric feature (string length) rather than feeding raw text paths as features. Thi |
| 2 | kaggle_p8 | PASS | PASS | FAIL | The notebook trains a neural network on the Titanic dataset and evaluates its accuracy. It explicitly states that the model achieves over 80% accuracy on the task, which is better  |
| 2 | kaggle_p9 | FAIL | PASS | FAIL | The notebook shows a from-scratch deep learning model trained with gradient descent on small batches without explicit regularization or early stopping to prevent overfitting. This  |

### Direct vs VibeTest Qwen ex20

Disagreements: 31 / 75

| row | property | GT | direct | VibeTest | direct reason |
|---:|---|---|---|---|---|
| 0 | kaggle_p0 | PASS | FAIL | PASS | The repository's main notebook uses the Titanic dataset without explicit data splitting into train, validation, and test sets. Model training and hyperparameter tuning (learning ra |
| 0 | kaggle_p12 | FAIL | PASS | FAIL | Evaluation uses a fixed trained model and a deterministic forward prediction function with no evidence of stochastic elements such as dropout or random sampling during evaluation.  |
| 0 | kaggle_p13 | FAIL | PASS | FAIL | The model and inputs run only on CPU as indicated by the environment metadata and absence of any explicit device or CUDA commands. No implicit CPU<->GPU transfers occur. Dtype is n |
| 0 | kaggle_p15 | PASS | PASS | FAIL | The only explicit loops present are essential for network layer iteration and rely on efficient matrix operations inside. No loops are found that should be replaced by matrix opera |
| 0 | kaggle_p4 | FAIL | PASS | FAIL | The repository contains a training function that explicitly splits the data into training and validation sets. The training loop applies any loss calculation solely on the training |
| 0 | kaggle_p6 | FAIL | PASS | FAIL | The repository uses a fixed index-based split for evaluation and calculates accuracy as the metric; this matches the implicit claim and correct definition of the metric used. No ev |
| 0 | kaggle_p8 | FAIL | PASS | INCONCLUSIVE | The notebook trains a linear and deep learning model on the Titanic dataset, showing model accuracy over 80%. This accuracy exceeds typical majority class baseline (~60%) even thou |
| 0 | kaggle_p9 | FAIL | PASS | INCONCLUSIVE | The notebook is an educational deep learning example for the Titanic dataset, showing typical model training steps without explicit constraints preventing single or tiny batch over |
| 1 | kaggle_p1 | FAIL | PASS | FAIL | The notebook explicitly shows a full manual implementation of training where all parameters are updated by gradient descent without freezing or skipping any parameters. |
| 1 | kaggle_p10 | FAIL | PASS | INCONCLUSIVE | The repository contains code that computes and logs training loss at each epoch, enabling monitoring that training loss decreases and plateaus during training. This satisfies the p |
| 1 | kaggle_p3 | PASS | PASS | FAIL | No data augmentation is implemented in the repository. The only data processing shown is deterministic tabular data cleaning and feature engineering which applies equally. Hence, a |
| 1 | kaggle_p4 | FAIL | PASS | FAIL | The repository notebook balances the dataset by downsampling classes before training, with no use of weighted loss or sampler for class imbalance handling seen anywhere. The balanc |
| 1 | kaggle_p5 | FAIL | PASS | FAIL | The repository processes the Titanic dataset in a way that converts categorical string columns (such as Sex, Pclass, Embarked, and an added TicketPath column constructed from Ticke |
| 1 | kaggle_p6 | PASS | INCONCLUSIVE | PASS | The notebook showcases training and some accuracy reporting calls but lacks explicit information on the dataset splits used for metrics or the exact metric definitions claimed (e.g |
| 1 | kaggle_p7 | FAIL | FAIL | INCONCLUSIVE | The only file is a single-notebook copy of a known tutorial. It trains a model but does not contain any steps or code related to label randomization or testing accuracy drop upon l |
| 1 | kaggle_p9 | FAIL | PASS | FAIL | The repository contains a complete notebook that trains a deep neural net from scratch on a small balanced Titanic dataset, with manual gradient update steps and no explicit regula |
| 2 | kaggle_p13 | FAIL | PASS | FAIL | The notebook uses PyTorch tensors to implement a model from scratch but contains no code for moving models or data to a specific device nor for consistent dtype policies. There is  |
| 2 | kaggle_p15 | FAIL | PASS | FAIL | The code uses efficient matrix operations for neural net calculations (e.g., torch matrix multiplication @). Non-matrix Python loops appear only for non-numeric operations like str |
| 2 | kaggle_p2 | FAIL | PASS | FAIL | The notebook demonstrates proper loading and preprocessing of the tabular Titanic dataset including missing value imputation, categorical variable encoding, and handling skewed num |
| 2 | kaggle_p3 | FAIL | PASS | FAIL | The notebook preprocessing consists of deterministic transformations only, with no evidence of augmentation on training or any differentiation vs validation/test preprocessing. Thu |
