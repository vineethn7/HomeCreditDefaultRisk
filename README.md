# Write Up

## Home Credit Default Risk


*Team Members(Group 18):*
<br></br>
 

| *Sno* &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;     | *Name&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | **Email*  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   |
| :---        |    :----:   |          ---: |
| 1.   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;     |    Venkat Vineeth Ram Kumar Nayakanti&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  |  vnayakan@iu.edu &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|
| 2.  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;    | Chandra Kiran Bachhu   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;    | kbachhu@iu.edu &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  |
| 3.  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Jagadeesh Chitturi     &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   | jchittu@iu.edu    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  |
| 4. &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  | Sripal Reddy Nomula    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;    | srnomula@iu.edu   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   |

<br>

<img src="https://i.imgur.com/2GFOY0K.jpg" height="800" width="800"/>

### Overview
The course project is based on the Kaggle Competition on Home Credit Default Risk (HCDR). This project's purpose is to anticipate if a client will repay a loan. Home Credit uses a number of alternative data—including telco and transactional information—to estimate their clients' repayment ability in order to ensure that people who struggle to secure loans owing to weak or non-existent credit histories have a pleasant loan experience.

## Abstract
The purpose of this project is to create a machine learning model/deep learning model that can predict consumer behavior during loan repayment.

In the earlier phases we performed extensive EDA on all the datasets along with designing a baseline model for the project HCDR. Then we used Logistic Regression Model as the baseline model with imbalanced dataset and also tried to use resampling to get a balanced dataset. Further we performed Feature Engineering and Aggregation to get the top features for training and predicting the TARGET. We also made use of different classification algorithms like Support Vector Machines, Decision Trees,and boosting algorithnms like AdaBoost, and XGboost, and hyperparameter tune to achieve the best model in each type of algorithm. We used metrics like Accuracy, Precision Score, F1 Score, ROC_AUC score and train the HCDR dataset and predict the TARGET attribute and check the performance of each model. Further the results from Phase 3 include the best classifier XGBoost as the winner and we also got a pretty good held test AUC score and kaggle score.

In the Phase 4 we worked on building MLP's using Pytorch Lightning and try to see if they produce some extraordinary results. We have built three models using Pytorch. One is a basic Single Layer Neural network, second one being a two layer neural network with binary cross entropy and hinge loss as its combined loss function. We also tried another variant of MLP with a Sigmoid Function included in the Forward step of the Neural Network along with a loss function equivalent to sum of Cross Entropy and Mean Squared Error. 

For the Results of Phase 4 we see the MLP with Sigmoid and two hidden layers had a AUC Score of 0.5 which was not what we expected. We also tried it without Sigmoid and we had got an AUC Score of 0.97 which was pretty good with a test accuracy of 94.6% but the predictions it made were quite interesting. 

We observed a pattern to our predictions when compare to the actual values. Whenever the actual was 0 our prediction was a positive integer more than 0 and when the actual was 1 our prediction was a negtive integer. It was quite interesting to witness this. We tried to find out a reason for this.

## Project Description

Home Credit is an international non-bank financial institution, which primarily focuses on lending people money regardless of their credit history. Home credit groups aim to provide positive borrowing experience to customers, who do not bank on traditional sources for pertaining loans. Hence, Home Credit Group published a dataset on Kaggle website with the objective of identifying and solving unfair loan rejection.

The purpose of this project is to create a machine learning model that can predict consumer behavior during loan repayment. Our task in this phase is to create a pipeline to build a baseline machine learning model using Logistic Regression algorithm. The resultant model will be evaluated with various performance metrics in order to build a better model. Companies can be able to rely on the output of this model to identify if loan is at risk to default. The new model built would help companies to avoid losses and make significant profits and will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

The results of our machine learning pipelines will be measured using the follwing metrics;
* Confusion Matrix
* Accuracy Score
* Precision
* Recall
* F1 score
* AUC (Area Under ROC Curve)

The pipeline results will be compared and ranked using the appropriate measurements and the most efficient pipeline will be submitted to the HCDR Kaggle Competition.

*Workflow*

For this project, we are following the proposed workflow as mentioned below.

<img src="https://i.imgur.com/03qFIjs.jpg" width=600 height=350/>

Below is the sub pipeline for Numerical and Categorical Pipeline.

<img src="https://i.imgur.com/5mFlYs9.png"/>

## Data Files Overview and Description

The `HomeCredit_columns_description.csv` acts as a data dictioanry.

There are 7 different sources of data(1 primary tables and 6 secondary tables):

### Primary Tables
* application_train

        This Primary table includes the application information for each loan application at Home Credit in one row. 
        This row includes the target variable of whether or not the loan was repaid. We use this field as the basis to determine the feature importance. The target variable is binary in nature based since this is a classification problem.
        * ‘1' - client with payment difficulties: he/she had late payment more than N days on at least one of the first M installments of the loan in our sample
        * '0' - all other cases
        The number of columns are 122. The number of data entries are 307,511.
    
* application_test

        This table includes the application information for each loan application at Home Credit in one row. The features are the same as the train data but exclude the target variable
        The number of columns are 121. The number of data entries are 48,744.
        
### Secondary Tables
* Bureau

        This table contains data points for all client's previous credits provided by other financial institutions that were reported to the Credit Bureau. There is one row for each previous credit, meaning a many-to-one relationship with the primary table. We could join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 17.The number of data entries are 1,716,428.

* Bureau Balance

        This dataset has the monthly balance history of every previous credit reported to the Credit Bureau. There is one row for each monthly balance, meaning a many-to-one relationship with the Bureau table. We could join it with bureau table by using bureau's ID, SK_ID_BUREAU.
        The number of columns are 3. The number of data entries are 27,299,925

* Previous Application

        This table contains records for all previous applications for Home Credit loans of clients who have loans in our sample. There is one row for each previous application related to loans in our data sample. , meaning a many-to-one relationship with the primary table. We could join it with primary table by using current application ID, SK_ID_CURR.
        There are four types of contracts:
        a. Consumer loan(POS – Credit limit given to buy consumer goods)
        b. Cash loan(Client is given cash)
        c. Revolving loan(Credit)
        d. XNA (Contract type without values)
        The number of columns are 37. The number of data entries are 1,670,214

* POS CASH Balance

        This table includes a monthly balance snapshot of a previous point of sale or cash loan that the customer has at Home Credit. There is one row for each monthly balance, meaning a many-to-one relationship with the Previous Application table. We would join it with Previous Application table by using previous application ID, SK_ID_PREV, then join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 8. The number of data entries are 10,001,358.

* Credit Card Balance

        This table includes a monthly balance snapshot of previous credit cards the customer has with Home Credit. There is one row for each previous monthly balance, meaning a many-to-one relationship with the Previous Application table.We could join it with Previous Application table by using previous application ID, SK_ID_PREV, then join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 23. The number of data entries are 3,840,312
        
* Installments Payments

        This table includes previous repayments made or not made by the customer on credits issued by Home Credit. There is one row for each payment or missed payment, meaning a many-to-one relationship with the Previous Application table. We would join it with Previous Application table by using previous application ID, SK_ID_PREV, then join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 8 . The number of data entries are 13,605,401
        

Below image gives the relationship between the different datasets for this project.

<img src="https://i.imgur.com/2hN3kal.jpg"/>

The training and testing dataset can be seen as below:

<img src="https://i.imgur.com/UaILpoO.jpg"/>

Different datasets and sizes are - 

<img src="https://i.imgur.com/r3DTKVI.jpg"/>

Missing Data for Training data in application_train.csv

<img src="https://i.imgur.com/ZH9vGlw.jpg"/>

Missing Data for testing data in application_test.csv

<img src="https://i.imgur.com/zOcZ80m.jpg"/>
Data Summary for Train and Test data:

<img src="https://i.imgur.com/J5gEsmI.jpg"/>

## Phase 2 Tasks:
## EDA(Exploratory Data Analysis)

Exploratory data analysis is important to this project because it helps to understand the data and it allows us to get closer to the certainty that the future results will be valid, accurately interpreted, and applicable to the proposed solution.

In the phase-2 of our project EDA helped us to look at the summary statistics on each table and focussing on missing data, Outliers and aggregate functions such as mean, median etc and visual representation of features for better understanding of the data.

For identifying missing data we made use of categorical and numerical features. Specific features have been visualized based on their correlation values. The highly correlated features were used to plot the density to evaluate the distributions in comparison to the target variable. We used different plots such as countplot, heatmap, densityplot, catplot etc for visualizing our analysis.

Visual EDA (Exploratory Data Analysis) involves using graphical representations and visualizations to explore and understand the data. It includes techniques such as scatter plots, histograms, box plots, heatmaps, and correlation matrices, among others. The objective of visual EDA is to identify patterns, trends, and outliers in the data that may not be immediately apparent in statistical EDA. Visual EDA helps to uncover relationships between features and the target variable, as well as the distribution and spread of the data. It can also help identify missing values and anomalies in the data. By using visual EDA, data analysts can gain a better understanding of the data, identify potential issues or biases, and make informed decisions about the data preprocessing and modeling phases of the project.

# Metrics - Definition and Formula

For Metrics we are going to make use of the ROC Curve, AUC Score, F1 Score, Precision Score, Accuracy metric to check the performance of our model. 

1. ROC Curve: The ROC (Receiver Operating Characteristic) curve is a graphical representation of the trade-off between the true positive rate (TPR) and false positive rate (FPR) for different threshold values of a binary classifier. It is a useful tool for evaluating the performance of a classification model at various decision thresholds.

\begin{equation}
\mathrm{ROC\ Curve:\ }\mathrm{TPR}(t) = \frac{\mathrm{TP}(t)}{\mathrm{P}}, \mathrm{FPR}(t) = \frac{\mathrm{FP}(t)}{\mathrm{N}}
\end{equation}

2. AUC Score: The AUC (Area Under the ROC Curve) score is a metric that measures the overall performance of a binary classification model based on the ROC curve. It represents the area under the ROC curve and ranges from 0 to 1, where a value of 1 indicates perfect classification and a value of 0.5 indicates random guessing.

\begin{equation}
\mathrm{AUC\ Score:\ }\mathrm{AUC} = \int_{-\infty}^{\infty} \mathrm{TPR}(t) \mathrm{dFPR}(t)
\end{equation}

3. F1 Score: The F1 score is a metric that combines the precision and recall of a binary classification model. It is the harmonic mean of precision and recall, and ranges from 0 to 1, where a value of 1 indicates perfect precision and recall.

\begin{equation}
\mathrm{F1\ Score:\ }\mathrm{F1} = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
\end{equation}

4. Precision Score: The precision score is a metric that measures the proportion of true positives among all the positive predictions made by a binary classification model. It represents the model's ability to correctly identify positive cases, and ranges from 0 to 1, where a value of 1 indicates perfect precision.

\begin{equation}
\mathrm{Precision\ Score:\ }\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
\end{equation}

5. Accuracy Score: The accuracy score is a metric that measures the proportion of correct predictions made by a binary classification model. It represents the model's overall ability to correctly classify both positive and negative cases, and ranges from 0 to 1, where a value of 1 indicates perfect accuracy.

\begin{equation}
\mathrm{Accuracy\ Score:\ }\mathrm{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP}+\mathrm{FP}+\mathrm{TN}+\mathrm{FN}}
\end{equation}

- <u>Note</u>: 
    - TP is the number of true positives, 
    - FP is the number of false positives, 
    - TN is the number of true negatives, 
    - FN is the number of false negatives, 
    - P is the total number of positives, and 
    - N is the total number of negatives.

## Feature Engineering and transformers
Feature Engineering is important because it is directly reflected in the quality of the machine learning model, This is because in this phase new features are created, transformed, dropped based on aggregator functions such as max, min, mean, sum and count etc. 

* Including Custom domain knowledge based features
* Creating engineered aggregated features
* Experimental modelling of the data
* Validating Manual OHE
* Merging all datasets
* Drop Columns with Missing Values

Domain knowledge-based features, which assist increase a model's accuracy, are an important aspect of any feature engineering process. The first step was to determine which of these were applicable to each dataset. Credit card balance after payment based on due amount, application amount average, credit average, and other new custom features were among them. Available credit as a percentage of income, Annuity as a percentage of income, Annuity as a percentage of available credit are all examples of percentages.

The next stage was to find the numerical characteristics and aggregate them into mean, minimum, and maximum values. During the engineering phase, an effort was made to use label encoding for unique values greater than 5. However, to reduce the amount of code required to perform the same functionality, a design choice was taken to apply OHE at the pipeline level for specified highly correlated variables on the final merged dataset.

Extensive feature engineering was carried out by experimenting with several modeling techniques using primary, secondary, and tertiary tables before settling on an efficient strategy that used the least amount of memory. For Level 3 tables bureau balance, credit card installment, installment payments, and point of sale systems cash balance, the first attempt entailed developing engineered and aggregated features. This was then combined with Level 2 tables, such as prev application balance with credit card installment, installment payments, and point of sale systems cash balance, as well as aggregated features, to create prev application balance. Along with the core dataset application train, a flattened view comprising all of the aforementioned tables was integrated. As a result, there were a lot of redundant features that took up a lot of memory.

Attempt 2 involved creating custom and aggregated features for Level 3 tables and merging with level 2 tables based on the primary key provided, which was later “extended” to the level 1 tables based on the additional aggregated columns. This approach created less duplicates.

A train dataframe was created by merging the level3, level2, and level1 datasets. There were extra precautions made to verify that no columns had more than 50% of the data missing.

The characteristics were engineered and included in the model with modest divides to assist test the model, however the accuracy was low. However, for XGBoost, employing these combined features in conjunction with acceptable splits throughout the training face resulted in improved accuracy and reduced the risk of overfitting.
Label encoding for unique categorical values in all categorical fields, not just a few, will be the focus of future research and trials.

## Pipelines

## Baseline Pipeline

The first step in any data analysis project is to obtain and import the necessary data. In this case, the data will be downloaded from Kaggle and then imported into the appropriate programming environment, along with any required modules and packages.

After the data has been imported, it is important to determine its shape and size. This allows us to get a sense of the number of observations and features in the dataset, which is necessary for performing exploratory data analysis (EDA). During the EDA phase, we will look for anomalies such as missing values or duplicate data, and handle them using imputation or dropping the attribute if it has a lot of missing values in it. It is important to deal with missing data as it can skew the results and lead to inaccurate conclusions.

Once the data has been cleaned and prepared, we can perform feature extraction to preserve the information in the original dataset. Feature extraction is the process of selecting the most relevant features for a given problem or task. This helps to reduce the dimensionality of the dataset and improve the efficiency and accuracy of the machine learning algorithms used.

Next, we can train the data on different machine learning models, such as logistic regression, decision trees, neural networks, and KNN. It is important to optimize these models by hyperparameter tuning, which involves adjusting the settings and parameters of the algorithm to find the best configuration for the given problem.

Finally, we can test the models on various metrics such as F1 score, ROC/AUC, and confusion matrix to evaluate their accuracy and performance. Based on the results, we can decide on the best model for the given problem and use it to make predictions and draw conclusions. Overall, this process involves a combination of data preparation, feature extraction, and model selection and optimization to achieve the best possible results.

Logistic regression model is used as a baseline Model, since it's easy to implement yet provides great efficiency. Training a logistic regression model doesn't require high computation power. We also tuned the regularization, tolerance, and C hyper parameters for the Logistic regression model and compared the results with the baseline model. We used 5 fold cross fold validation with hyperparameters to tune the model and apply GridSearchCV function in Sklearn.

Below is the workflow for the model pipeline.

<img src="https://i.imgur.com/syKYJOo.jpg"/>

## Phase -3 Tasks

In Phase 2, we used the Logistic regression model as the baseline model since it doesn't take a lot of computing resources and was simple to execute. We also used customized logistic models with a balanced dataset to increase the predictiveness of our model. In phase 3, we did look at different classification models to see if we can improve our forecast. Our main focus is on boosting algorithms, which are believed to be extremely efficient and relatively fast. The modeling workflow for phase 2 is depicted in the diagram below. We experimented with Decision Trees, Support Vector Machines, AdaBoost and XGBoost in our research.We also planned for KNearestNeighbours but had to drop out that algorithm because the computation power it needs and the memory usage it needs for a large dataset like the HCDR.

## Machine Learning Algorithms

We plan to evaluate several classifiers that can predict and classify loan recipients into two categories: "defaulters" and "non-defaulters". 

In the context of loan repayment, it is essential to identify those borrowers who are likely to default on their loans, as it can have a significant impact on the lender's financial health. Therefore, to make informed decisions, lenders can use different techniques to classify borrowers into two categories: "defaulters" and "non-defaulters."

Some of the commonly used techniques for classification are Logistic Regression, Decision Trees, Boosting Algorithms, Neural Networks, and Support vector machines.

- Logistic Regression is a statistical method that predicts the probability of a binary outcome, such as defaulting or not defaulting on a loan, based on one or more predictor variables. It can also identify any anomalies in the data, which can be useful for detecting fraudulent activities.

- Decision Trees are a type of non-parametric supervised learning algorithm that can be used for both classification and regression tasks. It works by partitioning the data into smaller subsets based on the values of predictor variables, eventually creating a tree-like structure that allows for the classification of new data.


- XGBoost (Extreme Gradient Boosting) is a popular and powerful algorithm for supervised learning tasks such as classification and regression. It is an ensemble learning method that combines multiple weak models into a strong one by iteratively adding decision trees that correct the errors of the previous ones. XGBoost is known for its efficiency, scalability, and accuracy, as well as its ability to handle missing values, feature interactions, and large datasets.

- Support Vector Machines (SVM) is a widely used algorithm for classification and regression tasks, especially in binary classification problems. SVM tries to find the best hyperplane that separates the data into different classes while maximizing the margin, which is the distance between the hyperplane and the closest points of each class. SVM can handle non-linearly separable data by using kernel functions that map the original data into a higher-dimensional space where it becomes separable. SVM is also known for its ability to avoid overfitting and handle high-dimensional data.

- To determine the best classifier, it is essential to evaluate the performance of each method by comparing their accuracy, precision, recall, F1 score, and other relevant metrics. By selecting the best classifier, lenders can make more informed decisions and reduce the risk of loan default.

To evaluate the effectiveness and precision of the models created using the algorithms mentioned above, we can employ the following measures:

- F1 score: The F1-score is a widely used metric for comparing the performance of binary classifiers. It combines the precision and recall of a classifier into a single metric by taking their harmonic mean. The formula for F1 score is:

    F1 = 2 * (precision * recall) / (precision + recall)
    
    where precision is the ratio of true positive predictions to the total number of positive predictions, and recall is the ratio of true positive predictions to the total number of actual positive instances.

- Receiver Operating Characteristic Area Under the Curve (ROC AUC): ROC AUC is a popular metric for evaluating the performance of binary classifiers. It measures the area under the receiver operating characteristic (ROC) curve, which shows the trade-off between true positive rate and false positive rate at different classification thresholds. A perfect classifier would have an ROC AUC of 1. The formula for ROC AUC is:

    ROC AUC = ∫(0, 1) TPR(FPR)^-1 dFPR

    where TPR is the true positive rate and FPR is the false positive rate.

- Accuracy: Accuracy measures the percentage of correct predictions made by the classifier. The formula for accuracy is:

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    where TP is the number of true positives, TN is the number of true negatives, FP is the number of false positives, and FN is the number of false negatives.
    
- Confusion matrix is another important metric used to evaluate the performance of a classifier. It provides a tabular representation of the number of correct and incorrect predictions made by the classifier for each class. The confusion matrix is typically represented as a 2x2 matrix for binary classification problems, where the rows represent the actual class labels and the columns represent the predicted class labels. The four possible outcomes are:

    True Positive (TP): The classifier correctly predicted a positive instance.
    
    False Positive (FP): The classifier incorrectly predicted a positive instance.
    
    True Negative (TN): The classifier correctly predicted a negative instance.
    
    False Negative (FN): The classifier incorrectly predicted a negative instance.
    
    The confusion matrix can be used to calculate several other metrics, such as precision, recall, and specificity. For example, precision is the ratio of true positive predictions to the total number of positive predictions, and recall is the ratio of true positive predictions to the total number of actual positive instances. The formulas for precision, recall, and specificity are:

    Precision = TP / (TP + FP) 
    
    Recall = TP / (TP + FN) 
    
    Specificity = TN / (TN + FP) 
    
    Overall, the confusion matrix provides a more detailed view of the performance of a classifier compared to other metrics, as it takes into account both true positives and false positives as well as true negatives and false negatives.

Below is the reason for choosing the mentioned models.
* Decision Trees: Decision trees are a popular choice for machine learning tasks because they are easy to understand and interpret. Decision trees are also robust to noisy data and can handle both categorical and continuous data. They can be used for both classification and regression tasks.

* XGBoost is one of the quickest implementations of gradient boosted trees. XGBoost is designed to handle missing values internally. This is helpful because there are many, many hyperparameters to tune. XGBoost is the best classifier from all our experiments.

* SVM performs similar to logistic regression when linear separation and performs well with non-linear boundaries depending on the kernel used. SVM is susceptible to overfitting/training issues depending on the kernel. A more complex kernel would overfit the model. We tried to experiment with this algorithm but we failed to produce results as the algorithm did not converge at all. We ran the GridSearch with hyperparameters but even after running for 7.5hrs we were unable to get the results so we had to proceed with other models. Also we tried to use linear kernel to reduce the SVC model computation and make it simpler but it also took too much time and showed no convergence.

* AdaBoost is another classic boosting algorithm that works by combining a set of weak classifiers to create a stronger classifier. AdaBoost is known to be effective in handling high-dimensional datasets and can handle both numerical and categorical features. Instead of stopping at the SVM deadend we tried to experiment with AdaBoostClassifier. This is part of sklearn.ensemble and this did converge. However we were not able to achieve higher accuracy. The highest accuracy was the XGBoost classifier. 

Boosting algorithms can overfit if the number of trees is very large. We did two submission in Kaggle, one using Baseline and Voting Classifier. A Voting Classifier is a machine learning model that trains on an ensemble of various models and predicts an output based on their highest probability of chosen class as the output. 

## Hyperparameters Used

Below are the hyperparameters we used for training different models:

```
params_grid = {
        'Logistic Regression': {
            'penalty': ('l1', 'l2'),
            'tol': (0.0001, 0.00001), 
            'C': (10, 1, 0.1, 0.01),
        }
    ,
        'Decision Tree' : {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 12],
            'min_samples_split': [100, 500, 1000],
            'min_samples_leaf': [50, 75, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'splitter': ['best', 'random']
        }
    ,
        # 'Support Vector Machines' : {
        #     'kernel': ['linear'],     
        #     'degree': (4, 5),
        #     'C': ( 0.0001, 0.001),   #Low C - allow for misclassification
        #     'gamma':(0.01,0.1,1),  #Low gamma - high variance and low bias
        #     'max_iter':(1000,10000,100000)
        # }
        'AdaBoost' : {
            'n_estimators':[200,300],
            'learning_rate': [0.01,0.1],
        }
    ,
        'XGBoost':  {
            'max_depth': [3,5], # Lower helps with overfitting
            'n_estimators':[200,300],
            'learning_rate': [0.01,0.1],
            'colsample_bytree' : [0.2], 
        },                      #small numbers reduces accuracy but runs faster 
    }
```

### Best Parameters for All models

**Logistic Regression**

<img src="https://i.imgur.com/5GmhlHf.png" />

**Decision Tree**

<img src="https://i.imgur.com/idONzap.png" />

**AdaBoost Classifier**

<img src="https://i.imgur.com/LWyAZts.png" />

**XGBoost Classifier**

<img src="https://i.imgur.com/OG3bd5f.png" />

## Experimental results

**Traditional Models**

Below is the resulting table for the results on the given dataset.

<img src="https://i.imgur.com/cmVxdxT.png" />


## Feature Importance

We tried to validate the model using the feature importance.

**Decision Trees**

<img src="https://i.imgur.com/LEqgc5E.png" />

**AdaBoost Classifier**

<img src="https://i.imgur.com/jpYLuhu.png" />

**XGBoost Classifier**

<img src="https://i.imgur.com/5Ayz044.png" />

## Phase 4 Tasks

## Pipeline for Multi Layer Perceptron and NN's training

- Neural Networks are a subset of artificial intelligence that uses interconnected nodes to generate directed or undirected graphs over time. This allows them to display temporal dynamic behavior, making them well-suited for time-series data, such as financial transactions.

- We used Pytorch Lightning module to develop our models. We tried using the Lightning Data Module as well.

Below is the pipeline that depicts the flow of this Phase 4 tasks:

<img src ='https://i.imgur.com/t4gfYzJ.jpg'/>

## Neural Networks Architecture

For the First Model which is a Single Neural Network has -
Only one input layer with input_dim as the num of columns in train set.
And output dim was set to 1.
We used Linear Layer.

For the Second model which is an Multi layer network with one hidden layer of 64 dim. We used combination of Linear and Relu for this.
Here we also implemented binary cross entrop and hinge loss as a combined loss function.

For the Third model which is also a Multi layer neural network but has two hidden layers, starting with 128 dims and followed by 64 dims.
Initially we used withoud sigmoid function. But later experimented with sigmoid as well
Here the loss is a combination of CXE and MSE.

In this final phase of this project we implemented  three different MLP models using PyTorch Lightning for classification as mentioned below:
1. Single Layer Neural Network having a fully connected Linear network with binary cross entropy as loss function
2. Multi Layer Neural Network having hidden layers and uses CXE and hinge loss function
3.  Multi Layer Neural Network having CXE+MSE as loss function
 The models had different architectures and loss functions, and were trained using a balanced dataset created in the previous phase.
 
 Initially, all models were trained with 50 and then we increased to 150 epochs and ROC score. A trial and error method was used to determine the optimal number of hidden layers for the multi-layer models. we have used Adam optimizer for all models, and a fixed learning rate of 1e-3 was maintained. Overall, this phase focused on fine-tuning the models to achieve the best possible accuracy and performance.
 
 Architecture for single layer NN model:
 
 This single layer model consists of one input layer with a certain number of features and one output layer, with a certain number of dense layers in between. The loss is calculated using the Binary Cross Entropy Loss function and the model is optimized using the Adam Optimizer with a learning rate of 1e-3. The activation function used in this model is relu.
 Architecture for Multi Layer Neural Network having hidden layers model:
 The Multi Layer Neural Network with hidden layers uses CXE and hinge loss functions. This model comprises an input layer with a certain number of features and an output layer, with a number of dense layers in between. The loss function used is Binary Cross Entropy, and the model is optimized using the Adam Optimizer with a learning rate of 1e-2. The activation function utilized in this model is ReLU.
 Architecture for  Multi Layer Neural Network model having CXE+MSE Loss function:
 The multi-layer neural network model with CXE+MSE loss function comprises an input layer with a specific number of features and one output layer, along with a certain number of dense layers in between. The loss is calculated using a combination of Binary Cross Entropy and Mean Squared Error loss functions. The model is optimized using the Adam optimizer with a learning rate of 1e-3, and relu activation function is used in this model.


## Leakage Problem:
We have taken several measures to address the issue of data leakage in our project. For instance, we have utilized Cross-validation folds during the model training process and have allocated a specific portion of the training data to the validation set. Furthermore, we have taken care to drop the target variable while splitting the data, ensuring that the model is not trained on information that it will later use to make predictions. Additionally, we have used One-Hot Encoding (OHE) for categorical attributes and standard scaling for numerical attributes. We have also employed imputing techniques such as the most frequent method for categorical attributes and mean imputation for numerical attributes. All of these steps have helped us to minimize the risk of data leakage throughout the project. Finally, we have maintained a separate validation set throughout the creation and testing of the model. This set was held out until we completed the final model, and we performed a final stability check on the validation set. By taking all these steps, we have ensured that our pipeline does not suffer from any leakage problem and does not violate any cardinal sins of machine learning.

## Results and Discussion: 
Results and discussion are critical components of any data analysis project or research study. In our Home Credit Default Risk Analysis project, the results were obtained by running the denormalized datasets through machine learning models such as Logistic Regression, Decision Trees and other boosting models, which generated performance metrics such as Accuracy and AUC scores. The results were then submitted to Kaggle for public and private score evaluations.

We encountered several challenges during this project, including the large size of the dataset, which made handling and preprocessing the data difficult, as well as issues with modeling the data. Additionally, we experienced machine crashes while running the models. Even after switching from Jupyter notebook in docker to Colab and from Colab CPU to GPU we were having trouble to get results for the SVM model.

Phase -3 Results 

Despite these challenges, we were able to train four models using 259 features. The first model we used was Logistic Regression, which yielded a test accuracy of 0.92 and a test AUC of 0.76.

Based on the models discussed above, XGBoost stood out as the best predictive model with 75.37% ROC score and followed by Logistic regression and the worst performance by Multi layer neural network with 59.34% AUC score.

    * Logistic Regression : This model was chosen as the baseline model trained with both balanced and imbalanced dataset with feature engineering. The first model we used was Logistic Regression, which yielded a train accuracy of 92.03%, test accuracy of 91.95% and a test AUC of 76.46%.
    
    * XGBoost : By far this model resulted in the best model. Both in terms of timing and accuracy for the selected features and balanced dataset. The accuracy of the training and test are 88.47% and 80.12%. Test ROC under the curve is 75.47%.

    * AdaBoost Classifier: This model was chosen to try another boosting model and see if it produces better accuracy than the rest of the models. The train accuracy is 69.39% and test accuracy is 70.2% and the test AUC score is 76.85%.

    * Decision Tree : Our decision tree model, produced training accuracy of 70.36% and test accuracy of 65.48%. Test ROC score came out as 60.79%.

Phase -4 Results

For the Single layered NN we had got a test accuracy of 0.6812, where as for the second model with multi layers we have got a test accuracy of 0.5027. The loss had increased when we increased the hidden layers and used a sigmoid function. 

Below is the experiment log for the Neural Network experiments:

<img src='https://i.imgur.com/0vTNyLa.png'/>

We also have the tensorboard logs for each version of our experiment showing the curves for accuracies, loss values.

## Tensorboard Results/Logs

<img src='https://i.imgur.com/TaexEq1.png'/>
<img src='https://i.imgur.com/8ogxH27.png'/>
<img src='https://i.imgur.com/YJhiNVW.png'/>
    
## Problems faced

The problem encountered apart from the accuracy of the model include:

* An unstable platform for running Machine Learning Models and collaboration.
* Long running models and system crash was the one of the biggest challenge we faced during training the model.

## Gap Analysis

Based on the leaderboard for Phase 2 for HCDR Projects we see Group 6 and 8 have the best results when compared to the others in terms of performance and training time. Many have not posted their train times for the baseline models and their experiments. Although individual results like train accuracy for some models from the leaderboard is higher than most teams, the trade-off doesn't look good when we check with ROC. We think we have improved our model with Feature Aggregation in this Phase 3, which is evident in the AUC test scores and experiment log of all the models and experiments carried out. 

After Phase 3 Leaderboard we see a much more improvement in terms of accuracy and ROC score. We also see some teams having used some new kinds of models that we have not tried and we tried to recheck how and what parameters they used and try to learn from it.

## Conclusion

Our implementation using ML models to predict if an applicant will be able to repay a loan was successful. Extending from the phase-1's simple baseline model, data modelling with feature aggregation, feature engineering, and using various data preprocessing pipeline both increased & reduced efficiency of models. Models used for prediction were Logistic Regression , ensemble model approaches using gradient boosting, Xgboost, AdaBoost and SVM. In the current phase we did try to implement Multi layer neural network model using Pytorch.
Our best performing model was XGBoost with the best AUC score of 75.47%. The lowest performing model is the Decision Tree model with 60.79 %. In Phase 4 we can conclude that the experimentation with Neural Networks using Pytorch Lightning was a fun and learning challenge for us and this gave rise to some unexpected results when we started increasing the Neural Network layers. Our test ROC score for the MLP is 0.5. Further taking this learning we would want to get a deeper understanding od Neural Networks to improve this research.


## Kaggle Submission
Here are the screenshots of your best kaggle submission for logistic regression baseline model and with all other models included using the voting classifier.   
In Phase 4 we also submitted a Kaggle submission for the MLP model we trained.

## Phase Leader Plan

| Phase | Contributor | Contribution Details |
| --- | --- | --- |
| Phase 1 | Vineeth | Meeting Scheduling, Data files overview, Metrics ||
| Phase 1 | Jagadeesh | Phase Leader |
| Phase 1 | Jagadeesh | ML Algorithms to be used, Pipeline |
| Phase 1 | Chandra Kiran | Describing data, Block model of Pipeline |
| Phase 1 | Sripal | Planning Credit assignment, Git repo creation |
| Phase 2 | Vineeth | Pipeline Coding, Running some experimental pipelines ||
| Phase 2 | Chandra Kiran | Phase Leader |
| Phase 2 | Jagadeesh | Making comparision, create presentation|
| Phase 2 | Chandra Kiran | Planning Credit assignment, making notes of results from experiments |
| Phase 2 | Sripal | Exploratory Data analysis, Decide which slides to be presented by each member for video presentation  |
| Phase 3 | Vineeth | Implementing pipeline, Hyperparameter Tuning, Recording results ||
| Phase 3 | Vineeth | Phase Leader |
| Phase 3 | Jagadeesh | Syncing the notebook |
| Phase 3 | Chandra Kiran | Planning credit assignment |
| Phase 3 | Sripal | Video Creation |
| Phase 4 | Vineeth | Building model and training the model ||
| Phase 4 | Sripal | Phase Leader |
| Phase 4 | Jagadeesh | Define a loss function, Video Presentation planning |
| Phase 4 | Chandra Kiran | Creating final repository on Github, Planning credit assignment |
| Phase 4 | Sripal | Work on appearance of the final notebook and final project presentation video |

## Gantt Chart for Phases 1 through 4

<img src="https://i.imgur.com/Oh0dI16.jpg"/>

## Credit Assignment Plan Phase 4


| Name | Task | Task Details | Effors(hr) |
| --- | --- | --- | --- |
| Sripal |  Phase Leader and Group Meeting | Lead the tasks for the phase and Project Discussions|  4 ||
|  | Work on final apperance of notebook | Sync the submission notebook and maintain neat structure of notebook |  2 | 
|  | Final project presentation video | Create a video presentation and make sure it has all the requirementse | 2 |
|  | Submission of phase 4 | Gather all files and submit phase 4 | 2 |
| Chandra Kiran |  Planning Credit assignment | create credit assignment |  3 |
|  | Create Repository | Creating Final repository on Github | 2 |
|  | Group Meeting | project discussions | 4 |
| Jagadeesh | Define loss function |  Implement CXE, hinge, MSE |  3 |
|  | Group Meeting | Project Discussions | 4 |
| Vineeth | Building model and training the model | Implement neural networks and multilayer perceptrons |  7 |
|  | Group Meeting | Project Discussions | 4 |
