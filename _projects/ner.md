---
layout: page
title: Named Entity Recognition
description: Natural Language Processing
img: assets/img/ner/cover.png
category: Deep Learning
---

### Overview
This project implements and compares BiLSTM-based models for Named Entity Recognition (NER), trained on the CoNLL-2003 dataset. The work demonstrates comprehensive preprocessing, model design, training strategies, and performance evaluation for identifying named entities in text.

### Repository
{% if site.data.repositories.github_repos %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% if repo.name == "NamedEntityRecognition" %}
      {% include repository/repo.html repository=repo %}
    {% endif %}
  {% endfor %}
</div>
{% endif %}
- Technologies Used: PyTorch, Python, NumPy, Pandas, Scikit-learn, GloVe Embeddings
- Dataset: CoNLL-2003 Shared Task Dataset
- GitHub Repository: [NagaHarshita/NamedEntityRecognition](https://github.com/NagaHarshita/NamedEntityRecognition)

### Dataset & Preprocessing
The project utilizes the CoNLL-2003 dataset, a standard benchmark for Named Entity Recognition tasks. The dataset preprocessing includes:

- Data Split: train, dev, and test sets
- Cleaning: Removal of empty lines and normalization
- Tokenization: Converting words and tags to indexed sequences
- Entity Types: Person (PER), Organization (ORG), Location (LOC), and Miscellaneous (MISC)


### Model Architecture

BiLSTM Base Model
```python 
BLSTM(
    embedding: Embedding(23700, 100)
    lstm: LSTM(100, 256, dropout=0.33, bidirectional=True)
    linear: Linear(512 → 128)
    elu: ELU(alpha=1.0)
    classifier: Linear(128 → 10)
)
```


BiLSTM with GloVe Embeddings
```python
BLSTM_Glove(
    embedding: Embedding(400000, 100)
    lstm: LSTM(100, 256, dropout=0.33, bidirectional=True)
    linear: Linear(512 → 128)
    elu: ELU(alpha=1.0)
    classifier: Linear(128 → 9)
)
```

### Training Configuration
The models were trained with carefully tuned hyperparameters:

- Optimizer: SGD with learning rate 0.8
- Learning Rate Scheduler: Dynamic adjustment during training
- Epochs: 100
- Batch Size: 32
- Loss Function: Weighted cross-entropy to handle class imbalance
- Padding Strategy: ignore_index = -1 for padded values


### Results & Performance
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>F1 Score</th>
                        <th>LOC F1</th>
                        <th>MISC F1</th>
                        <th>ORG F1</th>
                        <th>PER F1</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>BiLSTM Base</strong></td>
                        <td>93.65%</td>
                        <td>65.15</td>
                        <td>79.93</td>
                        <td>68.67</td>
                        <td>58.87</td>
                        <td>52.81</td>
                    </tr>
                    <tr>
                        <td><strong>BiLSTM + GloVe</strong></td>
                        <td>93.31%</td>
                        <td>63.54</td>
                        <td>79.58</td>
                        <td>66.42</td>
                        <td>55.82</td>
                        <td>50.82</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>


### Key Insights
#### Performance Analysis:

- Both models achieve high accuracy (>93%) on token-level classification
- Location entities show the best F1 scores (~80%), indicating clearer patterns
- Person and Organization entities present greater challenges due to contextual complexity
- Class imbalance significantly impacts performance, especially for PER and ORG entities

#### Technical Learnings:

- GloVe embeddings provide rich semantic representations but require careful tuning
- Proper padding and batch handling are crucial for consistent training
- Weighted loss functions effectively address class imbalance issues
- Bidirectional context significantly improves entity boundary detection


