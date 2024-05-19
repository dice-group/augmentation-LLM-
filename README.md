
# Data Augmentation using Large Language Models (LLMs) for Relation Extraction (RE)

This repository contains code for our paper titel: 'Enhancing Relation Extraction Through Augmented Data: Large Language Models Unleashed' The paper is accepted at [NLDB2024](https://nldb2024.di.unito.it/). Both the augmented data and the original dataset are available.
All the code is self-explanatory, and we have provided it in notebook format for easy understanding. Most of the cells contain instructions.

### Prerequisites

All the packages used in our experiments are easy to install with the `pip` command. If you encounter any issues related to packages or code execution, please create an issue, and we will be happy to assist and resolve the problem.


### Augmented Data

The augmented data from our experiments is available below:
- The "FewRel" directory contains all the augmented data generated using both the large language model and the rule-based approach.
- The "NYT-FB" directory contains data organized according to the different schemes explained in the paper.


| *Dataset*   | *Augmented Data*  |
|-------------|-----------|
|FewRel|[Download](augmented_data/FewRel/)|
|NYT-FB|[Download](augmented_data/NYT/)|

### Augmentation using Llama

We provide complete code in the notebook for easy execution and step-by-step understanding. By default, we've set Llama-2-7B version, which is easy to run. Users can easily change it to another version in the model name cell.

[Llama-Based Augmentation](promptllama.ipynb)

### Augmentation using Falcon

Similar to Llama, the code for augmenting data using Falcon is in the following notebook.

[Falcon-Based Augmentation](falconprompt.ipynb)

### Rule-Based Approach and Discussion Section

The following notebook contains code for generating rule-based augmentation following the same approach as we followed for the two language models.

[Rule-Based Augmentation](ruelbasedDA.ipynb)

### Hyperparameter used in our experiments:
| Parameter               | Value              |
|-------------------------|--------------------|
| Learning Rate           | $2e-5$             |
| Batch Size              | 32                 |
| Number of Epochs        | 20                 |
| Loss Function           | Cross Entropy Loss|
| Optimization algorithm  | AdamW              |
| Dropout                 | 0.1                |
| Max Sequence length     | 128                |
| Tokenizer               | bert-base-uncased  |


### Selected Model 
The choosen model for evaluation and it's relvant code for training and evaluation is available in 

[Model](Utility.py)

### Datasets used

| *Dataset*   | *Download*  |
|-------------|-----------|
|FewRel|[Download](https://www.zhuhao.me/fewrel/)|
|NYT-FB|[Download](http://iesl.cs.umass.edu/riedel/ecml/)|

<hr>

### Authors

  * [Manzoor Ali](https://dice-research.org/ManzoorAli) (DICE, Paderborn University) 
  * [Muhammad Sohail Nisar](https://dice-research.org/MuhammadSohailNisar) (DICE, Paderborn University)
  * [Muhammad Saleem](https://sites.google.com/site/saleemsweb/) (DICE, Paderborn University) 
  * [Diego Moussallem](https://dice-research.org/DiegoMoussallem) (DICE, Paderborn University)
  * [Axel-Cyrille Ngonga Ngomo](https://dice-research.org/AxelCyrilleNgongaNgomo) (DICE, Paderborn University)

## License
The source code of this repo is published under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
