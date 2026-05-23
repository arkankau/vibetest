### ml-bugs/kaggle_diabetic

| Method                       | Response Rate | Precision | Recall | F1     | Avg Cost (USD) | TP | FP | FN  | TN | Fail Labels | Fail Predictions |
| ---------------------------- | ------------- | --------- | ------ | ------ | -------------- | -- | -- | --- | -- | ----------- | ---------------- |
| AT-openai/gpt-5-mini         | 0.7013        | 0.5486    | 0.6038 | 0.5749 | 0.033792       | 96 | 79 | 63  | 65 | 159         | 175              |
| AT-openai/gpt-5-mini-dynamic | 0.6880        | 0.5600    | 0.5283 | 0.5437 | 0.023888       | 84 | 66 | 75  | 86 | 159         | 150              |
| codex-openai/gpt-5-mini      | 0.0373        | 0.6667    | 0.0503 | 0.0936 | 0.001163       | 8  | 4  | 151 | 2  | 159         | 12               |

### ml-bugs/kaggle_nlp

| Method                       | Response Rate | Precision | Recall | F1     | Avg Cost (USD) | TP  | FP | FN  | TN  | Fail Labels | Fail Predictions |
| ---------------------------- | ------------- | --------- | ------ | ------ | -------------- | --- | -- | --- | --- | ----------- | ---------------- |
| AT-openai/gpt-5-mini         | 0.7360        | 0.6763    | 0.7548 | 0.7134 | 0.024559       | 117 | 56 | 38  | 89  | 155         | 173              |
| AT-openai/gpt-5-mini-dynamic | 0.7893        | 0.6928    | 0.7419 | 0.7165 | 0.024187       | 115 | 51 | 40  | 111 | 155         | 166              |
| codex-openai/gpt-5-mini      | 0.0507        | 0.5789    | 0.0710 | 0.1264 | 0.001231       | 11  | 8  | 144 | 0   | 155         | 19               |

### ml-bugs/kaggle_titanic

| Method                       | Response Rate | Precision | Recall | F1     | Avg Cost (USD) | TP  | FP | FN  | TN  | Fail Labels | Fail Predictions |
| ---------------------------- | ------------- | --------- | ------ | ------ | -------------- | --- | -- | --- | --- | ----------- | ---------------- |
| AT-openai/gpt-5-mini         | 0.7840        | 0.7407    | 0.6316 | 0.6818 | 0.034928       | 120 | 42 | 70  | 108 | 190         | 162              |
| AT-openai/gpt-5-mini-dynamic | 0.9040        | 0.7639    | 0.5789 | 0.6587 | 0.027427       | 110 | 34 | 80  | 134 | 190         | 144              |
| codex-openai/gpt-5-mini      | 0.0773        | 0.4828    | 0.0737 | 0.1279 | 0.001048       | 14  | 15 | 176 | 0   | 190         | 29               |
| traincheck                   | 0.0000        | N/A       | 0.0000 | N/A    | N/A            | 0   | 0  | 190 | 0   | 190         | 0                |
