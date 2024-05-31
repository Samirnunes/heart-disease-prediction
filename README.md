# Heart Disease Prediction 

<p align="center">
    <img width="400" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/ilustration.jpg" alt="Material Bread logo">
<p>

Heart disease prediction project for CMC-16 (Data Science Practices) course.

## Running the Application

To run the application locally: `flask --app main.py run`

## Objectives

### The Project

The objective of this project is to create a machine learning model to predict the presence of heart disease in patients based in 13 patient's features associated to the clinical assessment moment, in order to assist doctors in clinical diagnosis. The target ("disease_degree" field) consists in an binary variable (0 - no disease; 1 - disease). Besides, the requirements include validating the model according to selected metrics and deploying it.

### Metrics

Regarding the metrics, it is proposed that the final model should achieve, on average, a minimum of 75% recall and 70% precision. Therefore, the model is expected to be primarily capable of avoiding false negatives. However, it should also avoid erroneously classifying too many patients as diseased.

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
    <img width="500" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/target_distribuition.png" alt="Material Bread logo">
<p>

### Preprocessing Pipeline

The `HdpDataPipeline` class was created to function as a data preprocessing pipeline for the problem. It encapsulates the following operations:

- Imputation of the mean for numerical variables;
- Imputation of the mode for categorical variables;
- Application of MinMaxScaler to scale the data to the [0, 1] range.

When the pipeline is applied to data after fitting on the training set, it performs the following additional operations along with the previous ones:

- Clipping the scaled feature values to the [0, 1] range;
- Checking if at least 50% of the variables are provided for the pipeline; if not, an error is raised, requesting the filling of more values.

Finally, during the model training (in the `HdpModelTrainer` class), oversampling is performed using SMOTE (Synthetic Minority Oversampling Technique) to balance the target classes.

### Model Choice: Validation

The validation of 5 models was carried out, which are listed below along with their parameters (using random_state = 100):

- `RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=3, max_leaf_nodes=10)`
- `DecisionTreeClassifier(random_state=random_state, max_depth=3, max_leaf_nodes=10)`
- `LogisticRegression()`
- `SVC(probability=True, random_state=random_state)`
- `XGBClassifier(random_state=random_state)`

The validation process involved checking the recall and precision histograms generated from 10 times the 10-fold cross-validation. The validation condition was that the mean minus one standard deviation must be at least greater than the minimum threshold considered (75% recall and 70% precision).

It was found that the two models that passed validation were `RandomForest` and `LogisticRegression`.

In the end, `LogisticRegression` was chosen as the model for deployment because it is less prone to overfitting, as also observed during the validation process. To analyze overfitting, the means of the metrics in the training folds were compared with the means of the metrics in the test fold.

- `LogisticRegression` histograms:

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/recalls_precisions_hist_logistic_regression.png" alt="Material Bread logo">
<p>

- `RandomForest` histograms:

<p align="center">
    <img width="500" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/recalls_precisions_hist_random_forest.png" alt="Material Bread logo">
<p>

### Model Test

After choosing `LogisticRegression` as the model, predictions were made on the test set, and the metrics were obtained, as shown below:

- Precision: 0.85
- Recall: 0.81

These results are consistent with the performed validation.

### Final Training and Deploy

Finally, the model was trained on all 920 observations of the dataset and deployed on a web page created using Flask, HTML, and CSS. Prediction through the model is done via an endpoint named predict_heart_disease, which receives the values of each feature and returns 0 or 1 for the disease prediction. Besides, it's possible to the doctor to give a feedback related to the prediction - he can select which prediction he thinks is correct, so we can compare with the model's output and use these feedbacks in the future to improve it.

The following image shows the application's user interface:

<p align="center">
    <img width="700" src="https://github.com/Samirnunes/heart-disease-prediction/blob/main/images/user_interface.png" alt="Material Bread logo">
<p>


