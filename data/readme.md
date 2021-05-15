# Datasets

## DBpedia-14

Dataset Summary
The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. They are listed in classes.txt. From each of thse 14 ontology classes, we randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000. There are 3 columns in the dataset (same for train and test splits), corresponding to class index (1 to 14), title and content. The title and content are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). There are no new lines in title or content.

Data for the DBpedia-14 dataset can be found here [here](https://huggingface.co/datasets/dbpedia_14).

## Jigsaw Unintended Bias in Toxicity Classification

### Detect toxicity across a diverse range of conversations

Data for the Toxicity Classification data can be found [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). Subsets of the dataset can be found in the [toxic_comments directory](https://github.com/JayThibs/Weak-Supervised-Learning-Case-Study/tree/main/data/toxic_comments).

Here’s the background: When the Conversation AI team first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.

For this dataset, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. Develop strategies to reduce unintended bias in machine learning models, and you'll help the Conversation AI team, and the entire industry, build models that work well for a wide range of conversations.
