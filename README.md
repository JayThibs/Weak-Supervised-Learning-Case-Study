# A Case Study on Weakly Supervised Learning

View our write-up of the project here: [A Case Study on Weakly Supervised Learning](https://docs.google.com/document/d/1oRHR_fUFEEYFOyHkR0sqv50XQFplEhcQhY0QrMw4z7s/edit?usp=sharing).

## Goal of the project

- Create a text data labeling service where the user inputs text data and receives a labeled dataset.
- Experiment with weak supervised learning and compare different approaches.

### Notebooks

* [01. Baseline with BERT on DBPedia-14](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/01_dbpedia_14_bert_classification_exploration.ipynb)
* [02. Distilling with Zero-Shot Classification on DBPedia-14](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/02_dbmedia_14_distilling_with_zero_shot_classification.ipynb)
* [03. Data Labeling DBPedia-14 with Snorkel](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/03_dbpedia_14_snorkel_dataset_labeling.ipynb)
* [04. Multi-Label Classification on Toxic Comments Dataset](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/04_transformers-multi-label-classification-toxicity.ipynb)
* [05. Toxcity Dataset Classifcation and Data Labeling with Snorkel](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/05_toxicity_classification_snorkel_dataset.ipynb)
* [06 Model Deployment in Azure Machine Learning Studio](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/06_AMLS_model_deployment.ipynb)

## Project Tree

```md
.
|-- ./pyproject.toml
|-- ./requirements
|   |-- ./requirements/dev.in
|   |-- ./requirements/dev.txt
|   |-- ./requirements/prod.in
|   `-- ./requirements/prod.txt
|-- ./setup.cfg
|-- ./project_proposal.md
|-- ./training
|   `-- ./training/__init__.py
|-- ./tasks
|   `-- ./tasks/lint.sh
|-- ./text_classifier
|   |-- ./text_classifier/__init__.py
|   |-- ./text_classifier/models
|   |   `-- ./text_classifier/models/__init__.py
|   |-- ./text_classifier/lit_models
|   |   `-- ./text_classifier/lit_models/__init__.py
|   `-- ./text_classifier/notebooks
|       |-- ./text_classifier/notebooks/01_dbpedia_14_bert_classification_exploration.ipynb
|       |-- ./text_classifier/notebooks/02_dbmedia_14_distilling_with_zero_shot_classification.ipynb
|       |-- ./text_classifier/notebooks/03_dbpedia_14_snorkel_dataset_labeling.ipynb
|       |-- ./text_classifier/notebooks/04_transformers-multi-label-classification-toxicity.ipynb
|       `-- ./text_classifier/notebooks/05_toxicity_classification_snorkel_dataset.ipynb
|-- ./Dockerfile
|-- ./data
|   |-- ./data/readme.md
|   `-- ./data/toxic_comments
|       |-- ./data/toxic_comments/test.csv
|       |-- ./data/toxic_comments/toxic_dev_200_examples.csv
|       |-- ./data/toxic_comments/toxic_test_630_examples.csv
|       |-- ./data/toxic_comments/toxic_train_2100_examples.csv
|       |-- ./data/toxic_comments/toxic_val_70_examples.csv
|       `-- ./data/toxic_comments/train.csv
|-- ./distill_classifier.py
|-- ./service.py
|-- ./test_request.json
|-- ./train_baseline_dbpedia_model.py
|-- ./05_toxicity_classification_snorkel_dataset.ipynb
|-- ./README.md
```

## Project Proposal

Find our project proposal **[here](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/project_proposal.md)**.
