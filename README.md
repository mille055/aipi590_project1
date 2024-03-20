# Automated Abdominal CT Protocoling Using Fine-Tuned LLM
#aipi590 project 1 

This project fine tunes a Mistral 7B Instruct model to protocol CT scans. 
<img src="assets/example_ct.jpg" width = "300" height = "200">

## Background
Performing the appropriate type of abdominal CT scan is important to high quality clinical care. The decisions of whether or not to use enteric contrast or IV contrast and the timing (phases) of imaging relative to IV contrast administration are encapsulated into specific protocols, which serve as recipes for performing the CT scan. Typically, this is performed entirely or in part by an individual, often a physician, and can be a time-intensive process given the large number of CT orders and occasionally vague statements of the indication for the examination necessitating review of the medical record. Automation of this process would allow the radiologist to focus time on other clinical duties such as interpreting scans. Large Language Models, which have shown a large number of promising uses in radiology, and may be well suited to incorporate information from the clinical chart into the protocol decision. While GPT-4 may be able to perform this task, HIPAA and privacy issues... 

## Methods

## Results

Custom scores on Test dataset:
```
Base Model (Mistral 7B Instruct) - 0.72 

Fine-Tuned Model - 0.95
```
For the performance on the protocol class alone, the accuracy for the base model was 73% and that for the fine-tuned model was 94%. The confusion matrix for the fine-tuned model is shown below:

<img src = "assets/confusion_matrix-2.png" width="400" height="400">


## References
1. Xavier BA, Chen PH, Natural Language Procesing fo Imaging Protocol Assignment, Journal of Digital Imaging (2022) 35:1120–1130.[Link](https://doi.org/10.1007/s10278-022-00633-8)
