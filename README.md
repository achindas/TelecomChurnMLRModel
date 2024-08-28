# Telecom Churn Prediction Case Study - Logistic Regression

## Table of Contents
* [Logistic Regression Overview](#logistic-regression-overview)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Approach for MLR](#approach-for-mlr)
* [Classification Outcome](#classification-outcome)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

## Logistic Regression Overview

**Logistic Regression** is a widely used statistical method for classification tasks, where the target variable is categorical and usually represents two classes (e.g., yes/no, 0/1, spam/ham). Unlike linear regression, which predicts continuous outcomes, logistic regression predicts the probability that a given input belongs to a particular class by modeling the relationship between the features and the **log odds** of the outcome.

## Key Concepts:

### 1. **Log Odds:**
Logistic Regression models the **log odds** of the probability of an event occurring. The **odds** are the ratio of the probability of an event happening to the probability of it not happening:

$$
	{Odds} = \frac{P(y=1 \mid \mathbf{x})}{1 - P(y=1 \mid \mathbf{x})}
$$

Taking the natural logarithm of the odds gives us the **log odds** (also known as the **logit**):

$$
\log\left(\frac{P(y=1 \mid \mathbf{x})}{1 - P(y=1 \mid \mathbf{x})}
\right)
$$

This logit is modeled as a linear combination of the input features:

$$
\log\left(\frac{P(y=1 \mid \mathbf{x})}{1 - P(y=1 \mid \mathbf{x})}
\right) = \mathbf{w}^T \mathbf{x} + b
$$

### 2. **Sigmoid Function:**
The log odds are transformed back into a probability using the **sigmoid** (or logistic) function, which maps any real-valued number to a value between 0 and 1:

$$
	{Sigmoid}(z) = \frac{1}{1 + e^{-z}}
$$

where $z = \mathbf{w}^T \mathbf{x} + b$. This allows us to interpret the output as a probability.

### 3. **Prediction:**
The probability that an input $\mathbf{x}$ belongs to the positive class (class 1) is:

$$
P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

To predict the class label, a threshold (typically 0.5) is used:
- Predict class 1 if $P(y = 1 \mid \mathbf{x}) > 0.5$,
- Otherwise, predict class 0.

### 4. **Confusion Matrix:**
A **Confusion Matrix** is a performance measurement tool used to evaluate the predictions of a classification model. It contains four key metrics:
- **True Positives (TP)**: Correctly predicted positive cases,
- **False Positives (FP)**: Incorrectly predicted positive cases (Type I error),
- **True Negatives (TN)**: Correctly predicted negative cases,
- **False Negatives (FN)**: Incorrectly predicted negative cases (Type II error).

By examining the confusion matrix, you can determine the overall effectiveness of the classifier and evaluate metrics such as **accuracy**, **precision**, **recall/ sensitivity**, **specificity** and **F1-score**.

### 5. **ROC Curve and AUC:**
The **Receiver Operating Characteristic (ROC) Curve** is a graphical representation of a model's ability to distinguish between the positive and negative classes across various threshold settings. The ROC curve plots **True Positive Rate (TPR)** (Recall) against **False Positive Rate (FPR)**:

$$
	{TPR} = \frac{TP}{TP + FN}
$$
$$
	{FPR} = \frac{FP}{FP + TN}
$$

- The **Area Under the Curve (AUC)** is a single scalar value summarizing the performance of the classifier. An AUC close to 1 indicates a very good classifier, while an AUC close to 0.5 indicates poor classification.

### 6. **Choosing the Optimal Threshold:**
In Logistic Regression, the threshold (commonly 0.5) used to classify predictions can significantly impact the performance of the model. However, this threshold may not always be optimal for all use cases, especially when the classes are imbalanced. To choose the best threshold, we can use the **ROC Curve** to find the point that maximizes both TPR and minimizes FPR.

**Multivariate Logistic Regression (MLR)** has been utilized in this case study in a step-by-step manner to understand, analyse, transform and model the data provided for the analysis. The approach described here represent the practical process utilised in industry to predict categorical target parameters for business.


## Problem Statement

A telecom firm has collected data of all its customers. The main types of attributes are:

* Demographics (age, gender etc.)
* Services availed (internet packs purchased, special offers taken etc.)
* Expenses (amount of recharge done per month etc.)

Based on all this past information, Telco Firm wants to build a model which will predict whether a particular customer will churn or not, i.e. whether they will switch to a different service provider or not. So the variable of interest, i.e. the target variable here is ‘Churn’ which will tell us whether or not a particular customer has churned. It is a binary variable - '1' means that the customer has churned and '0' means the customer has not churned.


## Technologies Used

Python Jupyter Notebook with Numpy, Pandas, Matplotlib and Seaborn libraries are used to prepare, analyse and visualise data. The following MLR specific libraries have been used in the case study:

- scikit-learn
- statsmodels


## Approach for MLR

A multi-step approach is used across Analyse, Build and Evaluate phases for this modelling exercise:

1. Import Data
2. Clean & Transform Data
3. Perform Data Analysis
4. Preprare Data for Modeling
5. Build Model
6. Evaluate Model Quality
7. Make Predictions on the Test Set
8. Conclusion

Some distinguishing processes in this approach include,

- Conversion of multi-level categorical variables (variables having more than two unique categorical values) to multiple dummy variables with 0 & 1 values

- Perform correlation data analysis using `Heatmap`, identify highly coorelated features and drop them from dataset

- Build the initial model with most appropriate variable(s) using training data set. If there are many variables, apply automated `Recursive Feature Elimination (RFE)` technique to narrow down on desired features faster

- Further fine-tune the model by reducing collinearity within the independent variables using `Variance Inflation Factor (VIF)` computation and eliminating collinear variables

- Evaluate initial model performance that was built using a probability threshold of 0.5. Assess the classification metrics like `sensitivity, specificity, accuracy` etc.

- In case the metrics are not at reasonable levels, create `ROC Curve` to assess if `area under curve` indicates a descent model performance. If so, then plot sensitivity and specificity at various thresholds to determine optimal probability threshold 

- Perform prediction using `Test` data set by scaling it using previously established `Scaler`

- Evaluate if the `Accuracy` score is still about the same with good sensitivity, specificity, precision scores on `test` data set also

- Finally explain the model in business terms for easy comprehension of Business Users

## Classification Outcome

* The logistic regression model assigns a probability score to each customer, indicating how likely they are to churn

* At first, classes were assigned to all the customers in the train data set using a probability cutoff of 0.5

* The model thus made, was very accurate (Accuracy about 80%), but it had a very low sensitivity (approx. 53%).

* Hence, 0.5 as the cutoff for probability was not appropriate for the model. So we tried with the cutoff, until we got the most useful model.

* Thus, a different cutoff was arrived at, i.e. 0.3, which resulted in a model with slightly lower accuracy (approx. 77%), but a much better sensitivity (approx. 78%).

* Using test data, the scores for **Sensitivity** (0.72) and **Specificity** (0.749) are found to be very similar to Train Scores indicating that we have a stable model that's not overfitting the training datasets.

<BR>
The probability of churn is defined by the equation:

$$
	{Probability}(P) = \frac{1}{1 + e^{-z}}
$$

where $z$ in our model is defined as follows:

$z = -1.658 - 0.943  \times  tenure + 0.346 \times PaperlessBilling + 0.46 \times SeniorCitizen - 0.722 \times Contract\_One\_year - 1.3 \times Contract\_Two\_year - 0.388 \times PaymentMethod\_Credit\_card - 0.331 \times PaymentMethod\_Mailed\_check + 0.805 \times InternetService\_Fiber - 0.973 \times InternetService\_No + 0.21 \times MultipleLines\_Yes - 0.405 \times TechSupport\_Yes + 0.34 \times StreamingTV\_Yes + 0.243 \times StreamingMovies\_Yes$

Features with negative coefficients reduces `Churn` probability, while features with positive coefficients increases it.

The model features which are important to determine customer churn are:

1. Contract Years - Longer contracts reduce churn
2. Internet Service - Having internet service increases churn
3. Tenure - Higher tenure reduces churn
4. StreamingTV - Having StreamingTV increases churn
5. Senior Citizen - Senior citizens have higher churn


## Conclusion

Logistic regression is widely used in industry for its ability to model binary outcomes, such as predicting customer churn, fraud detection, or loan default. It provides interpretable results by estimating the probability of a specific event occurring based on various input features, making it easier to understand the impact of different factors. Its simplicity and effectiveness in classification problems make it a preferred choice across finance, marketing, healthcare, and other sectors for making informed business decisions.


## Acknowledgements

This case study has been developed as part of Post Graduate Diploma Program on Machine Learning and AI, offered jointly by Indian Institute of Information Technology, Bangalore (IIIT-B) and upGrad.