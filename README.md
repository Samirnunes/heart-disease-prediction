# Heart Disease Prediction 

Heart disease prediction for CMC-16 (Data Science Practices) course. The goal is to predict the presence of heart disease in patients based in 13 patient's features associated to the clinical assessment moment. The target ("disease_degree" field) consists in an binary variable (0 - no disease; 1 - disease).

## Data

### Source

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

## Results

The notebook `data_analysis.ipynb` shows an initial data analysis used to understanding the basic about the dataset: the univariate distribuitions and the correlation between each variable and the target. 

For example, the correlations heatmap is showed below:

<p align="center">
    <img width="700" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/correlations.png" alt="Material Bread logo">
<p>
