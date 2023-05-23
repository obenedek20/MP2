# Data Instruction
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.

During the entire course of the pandemic, one of the main problems that healthcare providers have faced is the shortage of medical resources and a proper plan to efficiently distribute them. In these tough times, being able to predict what kind of resource an individual might require at the time of being tested positive or even before that will be of immense help to the authorities as they would be able to procure and arrange for the resources necessary to save the life of that patient.

## Content
The dataset was provided by the Mexican government. This dataset contains an enormous number of anonymized patient-related information including pre-conditions. The raw dataset consists of 21 unique features and 1,048,576 unique patients.

In the Boolean features, 1 means "yes" and 2 means "no". 

## DataSet Description

* **SEX**: 1 for female and 2 for male.
* **AGE**: of the patient.
* **CLASIFFICATION_FINAL**: covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
* **PATIENT_TYPE**: type of care the patient received in the unit. 1 for returned home and 2 for hospitalization.
* **PNEUMONIA**: whether the patient already have air sacs inflammation or not.
* **DIABETES**: whether the patient has diabetes or not.
* **COPD**: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
* **PREGNANT**: whether the patient is pregnant or not.
* **ASTHMA**: whether the patient has asthma or not.
* **INMSUPR**: whether the patient is immunosuppressed or not.
* **CARDIOVASCULAR**: whether the patient has heart or blood vessels related disease.
* **RENAL_CHRONIC**: whether the patient has chronic renal disease or not.
* **OTHER_DISEASE**: whether the patient has other disease or not.
* **OBESITY**: whether the patient is obese or not.
* **TOBACCO**: whether the patient is a tobacco user.
* **USMER**: Indicates whether the patient treated medical units of the first, second or third level.
* **MEDICAL_UNIT**: type of institution of the National Health System that provided the care.
* **HIPERTENSION**: whether the patient has hypertension or not.
* **INTUBED**: whether the patient was connected to the ventilator.
* **ICU**: Indicates whether the patient had been admitted to an Intensive Care Unit.
* **DATE_DIED**: If the patient died indicate the date of death, and 9999-99-99 otherwise.


## Proprocess

For the purpose of this project, we have done the following preprocessing steps:

1. change **DATE_DIED** to 1 and 2, 1 means the patient is dead and 2 means the patient is alive. Then remove the original **DATE_DIED** column and add a new column **LABEL**.
2. Remove duplicated rows from the original dataset.
3. Randomly sample rows from the original dataset for our experiments. Training set contains 3500 rows and dev/test set contains 1500 rows separately.

The goal of our project is to predict whether the patient will die based on the other features.