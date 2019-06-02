# Titanic-Data-Science
Titanic Data Science Project
===================
In this write-up, I will detail the why, what, and how of my first Data Science project. I tackled this project to apply what I have learned and gauge how powerful machine learning techniques can truly be.

My main objective was to analyze data about the Titanic's passengers from Kaggle (an online community for data scientists to find and publish data sets) and determine what sorts of people were more likely to survive. Below, I will recap my thought process and the steps I took towards my goal. 


----------

Data Description
-------------
I was given a training set and test set in the form of CSV files, filled with observed Titanic passengers, along with a variety of relevant information about them and whether they survived or not. The variables provided included: Passenger ID, Class, gender, age, number of parents and children, number of siblings and spouses, fare, cabin, and port of embarkation. 

My first thought was that some of these numerous variables were most likely largely inconsequential as to whether they survived or not. In addition, some observations were missing data. 

----------


Observations and Analysis
-------------------
**Observations**:
1. The Ticket variable contains a high ratio of duplicates and may not have a correlation with survival.
2.  The Cabin variable can be excluded as it is highly incomplete with many missing observations.
3.  PassengerID can also be excluded because it is not correlated to survival.
4.  The Name variable can be used to derive each observed person's title.

**Assumptions**
1.  Women and Children were more likely to have survived.
2.  Upper class passengers were more likely to survive.

After writing down these preliminary observations and assumptions, the next step was to verify their validity. 

*Pclass*
To observe the Pclass variable's significance, I compared all three classes and their corresponding survival rates, noting that there was a direct correlation between higher classes and higher survival rates.

*Gender*
Comparing male and female survival rates, I found that 74.2% of females survived while only 18.8% of males survived. This clearly indicates that females were much more likely to survive.

*SibSp*
SibSp was unique in that its correlation to survival was not linear, with values of 1 and 2 having the highest survival rates and 0 having the third highest.

*Parch*
This variable was similar to SibSp, with the only difference being that values of 3 had the highest survival rates. This could perhaps be due to a smaller number of observations, and also because people with more children were more likely to be chosen to survive.

*Embarked*
After comparing the three values of Embarked to one another, there was no clear correlation.

*Age*
I chose to analyze this variable in bands, meaning I grouped observations within a certain range into a category, then analyzed the categories as a whole. After this, I found a linear correlation between younger ages and higher survival rates.

At the end of all this, I concluded that age, gender, class, and size of family were all crucial elements in survival.

----------


Cleaning Data and Creating Model
-------------
After analyzing all the variables that were given to me, it was time to create a model based on what I learned. 

**Combining features into new features**
Because of the similarity between Parch and SibSP, I decided to combine them into one "Size of family" variable that would allow for a more robust model with less variables. After analyzing this new variable, however, I realized it would be best to instead create a "IsAlone" variable that would allow me to forgo Parch, SibSp, and SizeOfFamily. I also decided to combine Age and Class by multiplying the values together.

**Converting features into numerical values**
Because some variables had non-numerical values, it was necessary for them to be converted in order to fit the model. Gender was easy to change, and I simply made Female values 0 and Male values 1. Age was separated into bands, or groups, and given ordinal values with 0 being children and 4 being elders. Fare was also split into bands in a similar fashion as Age. I derived titles from each passenger's name, grouping them into Mr., Miss, Mrs. Master, and Rare as a blanket label for people that did not fit any of the other labels. These were also given ordinal values from 0-4. IsAlone was already a binary value, and Age*Class was numerical as well. Now that all of our variables were able to be fit into our model, it was simply a matter of using them to create different models.

**Modeling**
I split the training file into X_train and Y_Train, using all columns except the variable Survived for X_train, and only Survived for Y_train. X_test was derived from the testing file with PassengerID dropped. From here, it was only a matter of using imported types of predictive modeling algorithms. The models I used included: Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Naive Bayes classifier, Decision Tree, Random Forest, Perceptron, Artificial neural network, and Relevance Vector Machine. 

After running all models and recording their confidence scores, I found that the Random Forest and Decision Tree models were the most accurate, with both of them tying with 86.76% confidence scores.
