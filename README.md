# Data Augmentation using Large Language Models (LLMs) for Relation Extraction (RE)

This repository contains code for augmenting data for the Relation Extraction (RE) task using Large Language Models (LLMs) and a rule-based approach. Both the augmented data and the original dataset are available.

### Augmentation using Llama

We provide complete code in the notebook for easy execution and step-by-step understanding. By default, we've set Llama-2-7B version, which is easy to run. Users can easily change it to another version in the model name cell.

[Llama-Based Augmentation](promptllama.ipynb)

### Augmentation using Falcon

Similar to Llama, the code for augmenting data using Falcon is in the following notebook.

[Falcon-Based Augmentation](falconprompt.ipynb)

### Rule-Based Approach and Discussion Section

The following notebook contains code for generating rule-based augmentation following the same approach as we followed for the two language models.

[Rule-Based Augmentation](ruelbasedDA.ipynb)

### Prerequisites

Install the required packages for running the script using the following command:

```bash
pip install -r requirements.txt
```
### Selected Model 
The choosen model for evaluation and it's relvant code for training and evaluation is available in 

[Model](Utility.py)

### Datasets used

| *Dataset*   | *Download*  |
|-------------|-----------|
|FewRel|[Download](https://www.zhuhao.me/fewrel/)|
|NYT-FB|[Download](http://iesl.cs.umass.edu/riedel/ecml/)|

<hr>
