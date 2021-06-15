# A Case Study on Weakly Supervised Learning

View our write-up of the project here: [A Case Study on Weakly Supervised Learning](https://docs.google.com/document/d/1oRHR_fUFEEYFOyHkR0sqv50XQFplEhcQhY0QrMw4z7s/edit?usp=sharing).

Project was created for the Full Stack Deep Learning 2021 course. This project was chosen as one of the top projects from the course and presented at the project showcase.

## Goal of the project

- Create a text data labeling service where the user inputs text data and receives a labeled dataset.
- Experiment with weak supervised learning and compare different approaches.

### Notebooks

* [01. Baseline with BERT on DBPedia-14 (GitHub link)](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/01_dbpedia_14_bert_classification_exploration.ipynb) - [Colab version](https://colab.research.google.com/github/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/01_dbpedia_14_bert_classification_exploration.ipynb)
* [02. Distilling with Zero-Shot Classification on DBPedia-14 (GitHub link)](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/02_dbmedia_14_distilling_with_zero_shot_classification.ipynb)
* [03. Data Labeling DBPedia-14 with Snorkel (GitHub link)](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/03_dbpedia_14_snorkel_dataset_labeling.ipynb) - [Colab version](https://colab.research.google.com/github/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/03_dbpedia_14_snorkel_dataset_labeling.ipynb)
* [04. Multi-Label Classification on Toxic Comments Dataset (GitHub link)](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/04_transformers-multi-label-classification-toxicity.ipynb)
* [05. Toxicity Dataset Classifcation and Data Labeling with Snorkel (GitHub link)](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/05_toxicity_classification_snorkel_dataset.ipynb) - [Colab version](https://colab.research.google.com/github/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/05_toxicity_classification_snorkel_dataset.ipynb)
* [06. Model Deployment in Azure Machine Learning Studio (GitHub link)](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/06_AMLS_model_deployment.ipynb)

## How to use this Project

For using only the Snorkel approach to weak supervision, use the following notebooks in this order: [01](https://colab.research.google.com/github/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/01_dbpedia_14_bert_classification_exploration.ipynb), [03](https://colab.research.google.com/github/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/03_dbpedia_14_snorkel_dataset_labeling.ipynb), [05](https://colab.research.google.com/github/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/05_toxicity_classification_snorkel_dataset.ipynb), [06](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/06_AMLS_model_deployment.ipynb). 

For using only the model distillation approach to weak supervision, use the following notebooks int this order: [02](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/02_dbmedia_14_distilling_with_zero_shot_classification.ipynb), [04](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/text_classifier/notebooks/04_transformers-multi-label-classification-toxicity.ipynb).

For more information on how to deploy a Streamlit App of this project, please go to our [webapp directory](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/tree/main/webapp).

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
|-- ./tasks
|   `-- ./tasks/lint.sh
|-- ./Dockerfile
|-- ./distill_classifier.py
|-- ./service.py
|-- ./test_request.json
|-- ./train_baseline_dbpedia_model.py
|-- ./tree-md
|-- ./text_classifier
|   |-- ./text_classifier/__init__.py
|   |-- ./text_classifier/models
|   |   `-- ./text_classifier/models/__init__.py
|   |-- ./text_classifier/lit_models
|   |   `-- ./text_classifier/lit_models/__init__.py
|   `-- ./text_classifier/notebooks
|       |-- ./text_classifier/notebooks/01_dbpedia_14_bert_classification_exploration.ipynb
|       |-- ./text_classifier/notebooks/04_transformers-multi-label-classification-toxicity.ipynb
|       |-- ./text_classifier/notebooks/03_dbpedia_14_snorkel_dataset_labeling.ipynb
|       |-- ./text_classifier/notebooks/05_toxicity_classification_snorkel_dataset.ipynb
|       |-- ./text_classifier/notebooks/02_dbmedia_14_distilling_with_zero_shot_classification.ipynb
|       `-- ./text_classifier/notebooks/06_AMLS_model_deployment.ipynb
|-- ./data
|   |-- ./data/toxic_comments
|   |   |-- ./data/toxic_comments/test.csv
|   |   |-- ./data/toxic_comments/toxic_dev_200_examples.csv
|   |   |-- ./data/toxic_comments/toxic_test_630_examples.csv
|   |   |-- ./data/toxic_comments/toxic_train_2100_examples.csv
|   |   |-- ./data/toxic_comments/toxic_val_70_examples.csv
|   |   |-- ./data/toxic_comments/train.csv
|   |   |-- ./data/toxic_comments/toxicity_snorkel_dataset_3014ex.csv
|   |   `-- ./data/toxic_comments/toxicity_test_675ex.csv
|   `-- ./data/readme.md
|-- ./README.md
`-- ./webapp
    |-- ./webapp/Dockerfile
    |-- ./webapp/app.py
    |-- ./webapp/backend.py
    |-- ./webapp/demo_config.json
    |-- ./webapp/requirements.txt
    |-- ./webapp/run_webapp.sh
    |-- ./webapp/utils.py
    `-- ./webapp/README.md%
```

## Project Proposal

Find our project proposal **[here](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/blob/main/project_proposal.md)**.
