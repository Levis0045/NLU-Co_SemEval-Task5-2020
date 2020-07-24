# NLU-Co_SemEval-Task5-2020
Experimentations code used for SemEval-2020 Task 5: NLU/SVM based model apply to characterise and extract counterfactual items on raw data

# Resume

We try to solve the problem of classification of counterfactual statements and extraction of antecedents/consequences in raw data, by mobilizing on one hand Support Vector Machine (SVMs) and on the other hand Natural Language Understanding (NLU) infrastructures available on the market for conversational agents. 

# How to run these experiments

## Subtask1: counterfactual classification

### Dev environnement

Please use pipenv to install dependencies

```
pipenv --python=3.6
pipenv shell
pipenv install

```

### SVM methods: sklearn experiments

- Train the model with this script

```python

python3 scripts/task1-train_damien.py

```
- Evaluate the model with this script

```python

python3 scripts/task1-label_damien.py

```

### NLU methods : Rasa and Snips experiments

- Train Rasa, Snips, sklearn and fastext model with this script (uncomment the line at the end)

```python

python3 scripts/task1-train_elvis.py

```
- Evaluate Rasa, Snips, sklearn and fastext model with this script

```python

python3 scripts/task1-label_elvis.py

```

## Subtask2: antecedent and consequent extraction

- Train Rasa and Snips model with this script (uncomment the line at the end)

```python

python3 scripts/task2-train_elvis.py

```
- Evaluate Rasa and Snips model with this script

```python

python3 scripts/task2-label_elvis.py

```

# Publication's reference competition

```
@inproceedings{yang-2020-semeval-task5,
    title = "{S}em{E}val-2020 Task 5: Counterfactual Recognition",
    author = "Yang, Xiaoyu and Obadinma, Stephen and Zhao, Huasha  and Zhang, Qiong and Matwin, Stan and Zhu, Xiaodan", 
    booktitle = "Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval-2020)",
    year = "2020",
    address = "Barcelona, Spain",
}

```