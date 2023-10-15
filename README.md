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


## CLoser Look at Machine Learning Process

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