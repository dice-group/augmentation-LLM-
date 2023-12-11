# Data augmentation using LLMs for Relation Extraction (RE)
This repo contains code for augmentation of data for RE task using LLMs and also rule based approach. Also the augumented data and the orignal dataset are available.

### Augument using Llama
We have provided a complete code in notebook for easy to run and step by step understanding. By default we kept Llama-2-7B version which is easy to run. The user can easily change it to other version in the model name cell.
```
promptllama.ipynb
```

### Augument using Falcon
Similar to Llama the code for augmenting data using Falcon is in the following notebook.
```
falconprompt.ipynb
```
### Augument using Rule Based approach
The following notebook contain code for genearting rule based augmentation following the same approach as we followed for the two language models.
```
ruelbasedDA.ipynb
```

### Prerequisites
The required packages for running the script will be installed by running the following command:
```
 pip install -r requirements.txt
```


### Datasets used

| *Dataset*   | *Download*  |
|-------------|-----------|
|FewRel|[Download](https://www.zhuhao.me/fewrel/)|
|NYT-FB|[Download](http://iesl.cs.umass.edu/riedel/ecml/)|

<hr>
