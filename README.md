# Heart Disease Prediction 

<p align="center">
    <img width="300" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/ilustration.jpg" alt="Material Bread logo">
<p>

Heart disease prediction for CMC-16 (Data Science Practices) course. The goal is to predict the presence of heart disease in patients based in 13 patient's features associated to the clinical assessment moment. The target ("disease_degree" field) consists in an binary variable (0 - no disease; 1 - disease).

## Data

### Source

The database used for the project is available in the machine learning repository of the University of California, Irvine. The 4 databases available in the repository were merged: Cleveland, Hungary, Switzerland, and VA Long Beach, then adding up 920 rows of data.

- https://archive.ics.uci.edu/dataset/45/heart+disease

### Explanation

The columns are:

1. `age`: age in years
2. `sex`: sex (1 = male; 0 = female)
3. `cp`: chest pain type -- Value 1: typical angina -- Value 2: atypical angina -- Value 3: non-anginal pain -- Value 4: asymptomatic
4. `trestbps`: resting blood pressure (in mm Hg on admission to the hospital)
5. `chol`: serum cholestoral in mg/dl
 6. `fbs`: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7. `restecg`: resting electrocardiographic results -- Value 0: normal -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
8. `thalach`: maximum heart rate achieved
9. `exang`: exercise induced angina (1 = yes; 0 = no)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: the slope of the peak exercise ST segment -- Value 1: upsloping -- Value 2: flat -- Value 3: downsloping
12. `ca`: number of major vessels (0-3) colored by flourosopy
13. `thal`: 3 = normal; 6 = fixed defect; 7 = reversable defect

## Objectives

### The Project

The objective of this project is to create a machine learning model to predict the presence of heart disease in patients, in order to assist doctors in clinical diagnosis, based on exam results. The requirements include validating the model according to selected metrics and deploying it.

### Metrics

Regarding the metrics, it is proposed that the final model should achieve, on average, a minimum of 75% recall and 70% precision. Therefore, the model is expected to be primarily capable of avoiding false negatives. However, it should also avoid erroneously classifying too many patients as diseased.

## Tools

The Python language was used for software development. For data processing and manipulation, the Pandas library was utilized. The Scikit-Learn library was used for creating machine learning models. The Imblearn library was employed to address data balancing. Finally, the deployment was carried out using the Flask library.

## Results

### Data Analysis

The notebook `data_analysis.ipynb` shows an initial data analysis used to understanding the basic about the dataset: the univariate distributions and the correlation between each variable and the target. 

The correlations heatmap is showed below:

<p align="center">
    <img width="700" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/correlations.png" alt="Material Bread logo">
<p>

Besides, the target's class distribution is:

<p align="center">
    <img width="700" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/target_distribuition.png" alt="Material Bread logo">
<p>

