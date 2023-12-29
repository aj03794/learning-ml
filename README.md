# Understanding Machine Learning

## Course

- https://developers.google.com/machine-learning/crash-course
- This course is specifically on supervised machine learning

## Framing

- ML systems learn how to combine input to produce useful predictions on never-before-seen data

### Labels

- *Thing we're predicting* - the `y` variable in simple linear regression.
- Future price of wheat, type of animal in a picture, meaning of an audio clip, etc

### Feature

- Input variable - the `x` variable in simple linear regression
- Email spam detector example
  - Words in the email text
  - Sender's address
  - Time of day the email was sent
  - Email contains the phrase...


### Representation

#### Feature Engineering

- Process of extracting features from raw data
- Definition of a feature shouldn't change over time
- Features should not have extreme outliers
  - Filter out these outliers

##### Mapping Categorical Values

- Imagine that there is a feature with options that include
  - `{'Charleston Road', 'North Shoreline Boulevard', 'Shorebird Way', 'Rengstorff Avenue'}`
  - Since models cannot multiply strings by the learned weights, we convert these strings to numeric values
- We can accomplish this by defining a map from the feature values, this is called the **vocabulary** of possible values, to integers
- Since not every street in the world will appear in our dataset, we can group all other streets into a catch-all "other" category know as an **OOV (out-of-vocabulary) bucket**
- Using this approach, here's how we can map our street names to numbers:
  -  charleston road = 0
  -  north shoreline boulevard = 1
  -  shorebird way = 2
  -  rengstorff avenue = 3
  -  everything else = 4

###### One Hot Encoding

- BQ ML uses *one hot encoding* by default
  - https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-glm#category_encoding_method
  - Talks about feature transformations here: https://cloud.google.com/bigquery/docs/auto-preprocessing#one_hot_encoding
- One-hot encoding
  - https://developers.google.com/machine-learning/glossary/#one-hot-encoding
  - Represents categorical data as a vector
    - One element is set to 1
    - All other elements are set to 0
  - Commonly used to represent strings or identifiers that have a finite set of possible values

![Alt text](./images/23.png)

- Spare representation
  - Suppose you had 1M different street names in your data set that you wanted to include as values for a particular feature, explicitly creating a binary vector of 1M elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and coputation time when processing these vectors
  - In this case,common approach is to use a sparse representation in which only nonzero values are stored
  - https://cloud.google.com/blog/topics/developers-practitioners/sparse-features-support-in-bigquery


### Feature Crosses

#### Encoding Nonlinearity

- Rarely used in neural networks

### Classification

- Is something A or B
- There are multiple metrics to evaluate classification models

#### Evaluation Metrics: Accuracy

- Fraction of predictions we got right
  - All things that were predicted correctly divided by everything
  - `(true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)`
- In many cases, accuracy is a poor or misleading metric
  - Typical case includeds *class imbalance*, when positives or negatives are extremely rare
    - Imagine an ad click through rate and all of your features are "false" indicating that it wasn't clicked - model would only produce false
  - Imagine model accuracy is 99.99% accurate, but what you really care about is finding out the .01% of ads that are succcessful


##### Class-imbalanced Data Set

- Watch out for a **class-imbalanced data set**
- A dataset for a classification problem in which the total number of labels of each class differs significantly
- For example consider a binary classification whose two labels are divided as follows
  - 1,000,000 negative labels
  - 10 positive labels
  - The ratio of negative to positive labels is 100,000:1, so this is a class-imbalanced dataset
- Imagine another dataset where there is not a class-imbaalnced
  - 517 negative labels
  - 483 positive labels
  - The ratio of negative labels to positive labels is roughly 1
- This same problem can exist with mutli-class datasets


#### True Positives and False Positives

- For class imbalanced problems, useful to separate out different kinds of errors

![Alt text](./images/17.png)

#### Evaluation Metrics: Precision and Recall

- Precision
  - *What proportion of positive identifications was actually correct?*
  - `true positives / all positive predictions` = `true_positives/(true_positives + false_positives)`
    - When a model said "positive" class, was it right
    - Did the model cry "wolf" too often?
- Recall
  - *What proportion of actual positives was identified correctly*
  - `true positives / all actual positives` = `true_positives/(true_positives + false_negatives)`
    - Out of all possible positives, how many did the model correctly identify?
    - Did it miss any wolves?
- These two metrics can be in contention
  - If you want to be better at recall, you're going to be more aggressive about saying wolf (lowering the classification threshold)
    - Typically raises it, doesn't always
  - If you want to be more precise, you'll only say wolf when you're absolutely sure (raising the classification threshold)
    - Typically raises it, doesn't always
  - Good illustration of this - https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
  - Good quiz: https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-accuracy-precision-recall
- Both of these values are very important in model evaluation

#### ROC Curve

- ROC = receiver operating characteristic
- Imagine that you don't know the best classification threshold is going to be, still want to know if model is doing a good job
- We might want to evaluate our model across all possible classification thresholds
- It's a graph showing the performance of a classification model at all classification thresholds

![Alt text](./images/18.png)

- Curve plots two parameters
- **True Positive Rate (TPR)**
  - `true_positive_rate = (true_positive / (true_positive + false_negative))`
- **False Positive Rate (FPR)**
  - `false_positive_rate = (false_positive / (false_positive + true_negative))`
- An ROC curve plots TPR vs FPR at different classification thresholds
  - Lowering the classification threshold classifies more items as positive which increases both False Positives and True Positives

![Alt text](./image/21.png)
  
##### AUC

- AUC = area under the ROC curve
- Measures the 2D area underneath the entire ROC curve (integral calculus) from (0, 0) to (1, 1)
- If we pick a random positive and a random negative, what's the probability the model ranks them in the correct order - higher score to the positive and lower score to the negative
- Gives an aggregate measure of performance aggregated across all possible classification thresholds

![Alt text](./images/19.png)

- Provides an aggregate measure of performance across all possible classification thresholds
- One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example

![Alt text](./images/22.png)

- AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example
- AUC ranges in value from 0 to 1
  - A model whose predictions are 100% wrong has an AUC of 0.0, and one whose predictions are 100% correct has an AUC of 1.0
- AUC is desirable for 2 reasons
  - AUC is **scale-invariant** - measures how well predictions are ranked; rather than their absolute values
  - AUC is **classification-threshold-invariant** - measures quality of the model's predictions irrespective of what classification threshold is chosen
- Both these reasons come with caveats
  - **Scale invariance is not always desireable**
    - For example, sometimes we really do need well calibrated probability outputs, and AUC won't tell us about that
  - **Classification-threshold invariance is not always desireable**
    - In cases where there are wide disparities in the cost of false negatives vs false positives, it may be critical to minimize one type of classification error
      - For example, when doing email spam detection, you likely want to prioritize minimize false positive (een if that results in a significant increase of false negatives)
      - AOC isn't a useful metric for this type of  optimization

#### Prediction Bias

- Expected value should be actual
- If there is deviation, then we say the model has a *bias*
- Useful to know when there is something wrong with the model
  - Incomplete feature set, buggly pipeline, biases training sample
  - Having a bias of 0 doesn't necessarily mean our model is accurate
  - Look for bias in slices of data - this can guide improvements
- Example, if 1% of all emails are spam, then our model should predict, on average, that emails are 1% likely to be span

##### Bucketing and Prediction Bias

- Logistic regression predicts a value *between* 0 and 1, but all labeled examples are either exactly 0 or exactly 1
- This means when examining prediction bias, you cannot accurately determine the prediction bias on only 1 example
  - Need to examine the prediction bias on a "bucket" of examples
  - Predictin bias only makes sense when grouping enough examples together to be able to compare to a predicted value (for example, 0.392) to observed values (for example, 0.394)

#### Calibration Plots Show Bucketed Bias

#### Classification: Thresholding

- Logistic regression returns a probability - you can use the probability "as is" (for example, probability that the user will click on this ad is 0.00023) or convert the returned probability to a binary value
- A logistic regression model that returns 0.9995 for a particular email mesage is predicting that it is very likely to be spam
- In order to map a logisitic regression to a binary category, you must define a **classification threshold** (aka **decision threshold**)
  - If your example is defining whether an email is spam or not, above this threshold indicates spam, and a value below indicates not spam
- *Tuning* a threshold is different than tuning hyperparameters such as learning rate
  - Part of choosing a threshold is assessing how much you'll suffer for making a mistake
    - For example, labeling a non-spam message as spam is very bad, but labeling spam as non-spam is annyoing, but not impactful


#### Confusion Matrix

- It's a table that summarizes the number of correct and incorrect predictions that a classification model made
- **Ground truth** is the thing that actually happened - it's reality
  - For example consider a binary classification model that predicts whether a student in their first year of univesity will graduate within 6 years. Ground truth for this model is whether or no that student actually graduated within 6 years

- Example is with crying wolf

![Alt text](./images/20.png)

- A **true positive** is an outcome where the model *correctly* predicts the *positive* class
- A **true negative** is an outcome where the model *correctly* predicts the *negative* class
- A **false positive** is an outcome where the model *incorrectly* predicts the *positive* class
- A **false negative** is an outcome where the model *incorrectly* predicts the *negative* class

#### Accuracy

### Regularization

- 

## Course

- https://app.pluralsight.com/library/courses/understanding-machine-learning

## What Is Machine Learning

### Getting Started

- Finds patterns in data
- Uses those patterns to predict the future
- Examples:
    - Detect credit card fraud
    - Determine whether a customer is likely to switch to a competitor
    - Deciding when to do preventive maintenance on a factory robot

**What does it mean to learn?**

- Learning requires:
    - identifying patterns
    - recognizing those patterns when you see them again
- This is what machine learning does

### Machine Learning in a Nutshell

![](./images/1.png)

- The model is code

### Why is Machine Learning so Popular?

- Doing machine learning well requires
    - Lots of data
        - We capture more and more data
    - Lots of compute power
        - We have the cloud
    - Effective machine learning algorithms
        - Researches have found what works and what doesn't

### The Ethics of Machine Learning

- What if data is biased?
    - Example: some sort of racial bias in the data will cause that same racial bias to exist in the model
- Models are generated by the machine learning process
    - Uses complex statistical processes
    - Can't just look at it to understand what it does
    - Can be hard to explain why the model is doing what it is doing

## The Machine Learning Process

### Getting Started

- Iterative
- Challenging
    - Can be working with very complex data
- Often rewarding

### Asking the Right Question

- Choosing what question to ask is the most important part of the process
- Do you have the right data to answer this question?
- Do you know how you'll measure success?

### The Machine Learning Process

- Choose data
    - Work a domain expert that knows a lot about credit card fraud
- Once data is ready, you can apply the learning algorithm to the data
- Result of learning algorithm is candidate model
    - Probably isn't the best
    - You produce several
- Deploy chosen model
- Applications can use the model

![](./images/2.png)

- Have to constantly update model because reality changes

![](./images/3.png)

### Examples

**Scenario: Detecting Credit Card Fraud**

![](./images/4.png)

**Scenario: Predicting Customer Churn**

![](./images/5.png)

**Common Use Cases**

- Recommandations
    - Think netflix shows
- Speech recognition
- Language translation
- Facial recognition
    - Ethical gray area


## Closer Look at Machine Learning Process

### Terminology

- Training data
    - Prepared data used to create a model
    - Creating a model is called *training* a model
- Supervised learning
    - The value you want to predict is in the training data
        - In credit card example, whether transaction was fraudalent or not was contained in each record
        - This data is *labeled*
        - More common than unsupervised learning
- Unsupervised learning
    - Value you want to predict is NOT in the training data
    - The data is not labeled

### Data Pre-Processing

- Could be relational, nosql, binary, etc
- Need to read the raw data into some `data preprocessing module(s)`
    - Provided by machine learning 
    - Lots of time spent getting data into right form for the module
- Create training data
    - Columns in `training data`  are called `features`
    - The last row - the `target value` is the value you are trying to predict
        - In the credit card fraud scenario, this would be whether the transaction as fraudalent or not

![](./images/6.png)

### Categorizing Machine Learning Problems

#### Regression

![](./images/7.png)

#### Classification

![](./images/8.png)

- In our example, classes would be whether the transaction was fraudalent or not
- You get a probability here, not a yes or no

#### Clustering

- We don't necessarily know wha we're looking for

![](./images/9.png)

### Styles of Machine Learning Algorithms

![](./images/10.png)

### Training and Testing a Model

- Create training data
- `Target Value` is part of the training data
- Choose the `features` that will be most predicitive of that `target value`
- How do we decide what `features` to use?
    - This is what data scientists are for
- How do we choose the right learning algorithm?
    - This is what data scientists are for

![](./images/11.png)

#### Testing a Model

- Input the remaining 25% into candidate model that was created previously

![](./images/12.png)

#### Improving a Model: Some Options

- Use different features
- Maybe the wrong dataset 
- Maybe algorithm is wrong
- Modify existing algorithm

### Using a Model

![](./images/13.png)

### Implementing Machine Learning

- Create custom models in R and Python using general ML packages
- Create custom models using more focused packages like TensorFlow
- Create custom models using cloud ML services like Amazom SageMaker
- Use predefined models like Azure Cognitive Services
    - Image
    - Speech
    - Recommendations


## Deep Learning

### Activation Functions

- Used to introduce non-linearity into the model
- This is important to model complex relationships - within our data there are likely non-linear relationships
- Where they are used depends on the type of algorithm
  - For example, in multiclass classification, activation functions are used in the hidden layers (using something like ReLU function) and output layers (like Softmax function)
  

#### Binary Classification

- Activation function is used to help decide whether the output is yes/no or true/false
- Activation functions are used in both the hidden layers and the output layers


##### Hidden Layers

- Common choices include Rectified Linear Unit(ReLU) and its variants like Leaky ReLU, Parametric ReLU, and Exponential Linear Unit


##### Output Layer

- Most common activation function in the output layer is particularly important as it converts the final layer's output into a format suitable for classification
- Most common choice is sigmoid function - maps output to probability score between 0 and 1

### Neural Networks

#### Binary Classification

