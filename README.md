# Automated Abdominal CT Protocoling Using Fine-Tuned LLM
#aipi590 project 1 

This project fine tunes a Mistral 7B Instruct model to protocol CT scans. 

<img src="assets/example_ct.jpg" width = "300" height = "200">

## Background
Performing the appropriate type of abdominal CT scan is important to high quality clinical care. The decisions of whether or not to use enteric contrast or IV contrast and the timing (phases) of imaging relative to IV contrast administration are encapsulated into specific protocols, which serve as recipes for performing the CT scan. Typically, this is performed entirely or in part by an individual, often a physician, and can be a time-intensive process given the large number of CT orders and occasionally vague statements of the indication for the examination necessitating review of the medical record. Automation of this process would allow the radiologist to focus time on other clinical duties such as interpreting scans. Large Language Models, which have shown a large number of promising uses in radiology, and may be well suited to incorporate information from the clinical chart into the protocol decision. While GPT-4 may be able to perform this task, HIPAA and privacy issues make data preclude releasing the sensitive data to the cloud. This project assesses the feasibility of using a fine-tuned open source local model, Mistral 7B Instruct, for this protocoling task.

Prior efforts have looked at other machine learning models in conjunction with NLP for text embeddings, achieving F1 scores for protocol selection to be in the range of 0.8-0.85 for various types of machine learning models including Random Forest, Gradient Boosted Tree, among others [1]. To our knowledge, there are no published reports of using an LLM for this task, although authors have speculated upon the potential for this use case [2]. 

## Methods
I used Mistral 7B Instruct-v0.2 as a base model and then fine-tuned over the training dataset. 
1. Training dataset - comprised of approximately 1200 exsamples of a typical order for an abdominal CT scan and some of the information accessible in the EHR when completing a protocoling task, including the serum creatinine (a marker of renal function- high creatinine above 2.0 mg/dL is a relative contraindication to giving IV contrast due to the risk of Contrast-Induced Nephropathy, unless the patient is on dialysis), the presence and severities of contrast allergy, and a brief clinical summary of 2-3 sentences. The order and prior CT protocol data are real clinical data from the EHR, whereas the other data were synethesized by GPT-4 (the clinical summary) or arbitrarily set. I performed the protocol task based on this set of inforation as the ground truth for comparison.
2. The prompt was comprised of a brief set of instructions followed by the clinical data. A typical prompt is given below:
   ```
   The task is to use the provided information for each patient to return the predicted protocol for the CT study in the form of a json object like this:
{"predicted_order": "CT abdomen pelvis with contrast", "predicted_protocol": "routine", "predicted_comments": ["oral contrast"]}
The response should be the json object and nothing else. 

'Order: CT chest abdomen pelvis with contrast with MIPS
Prior Order: CT abdomen pelvis without contrast
Reason for Exam: Metastatic melanoma, of uncertain site
Contrast Allergy: 0
Allergy severity:
Creatinine (mg/dL): 1.1
On Dialysis: 0 
Clinical Summary: The patient is a 45-year-old male with newly diagnosed melanoma (Clark level IV) arising from the left great toe. The scan is for initial staging purposes.’
```


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
