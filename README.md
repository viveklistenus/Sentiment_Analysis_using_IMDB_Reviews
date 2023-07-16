<!-- PROJECT TITLE -->
<h1 align="center">Sentiment Analysis using IMDb Reviews</h1>

<!-- HEADER -->
![thumbnail](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/edeb5568-1c06-491b-8849-e786f7fb4c75)

<!-- PROJECT DESCRIPTION -->
## <br>**➲ Project description**

Sentiment Analysis is a popular Natural Language Processing (NLP) task that aims to classify the sentiment of a given text as either positive, negative or neutral. In this project, IMDb reviews are used to train and evaluate a Sentiment Analysis model using the oneDAL toolkit from Intel.

The IMDb dataset consists of 5,000 movie reviews, which are labeled as either positive or negative. The oneDAL toolkit is used to preprocess the data and build a Naive Bayes model for sentiment analysis. The oneDAL library provides efficient and optimized algorithms for data preprocessing, feature engineering, and model training, which helps to achieve high accuracy in classification tasks.

The project involves several steps, including data cleaning, feature extraction, model training, and evaluation. First, the dataset is preprocessed using oneDAL algorithms, such as data normalization and feature scaling, to prepare it for training. Next, feature extraction techniques such as Bag-of-Words and TF-IDF are applied to transform the text data into numerical features.

After that, the model is trained using the oneDAL library, which provides a scalable and efficient implementation for large datasets. Finally, the model is evaluated on a test dataset to measure its accuracy and performance.

The results of the Sentiment Analysis using oneDAL on IMDb reviews project show that the model achieved high accuracy in classifying the sentiment of the reviews, with an accuracy score of over 80.3%. This project demonstrates the effectiveness of oneDAL library for NLP tasks and highlights its potential in building efficient and accurate machine learning models.



## NLP Pipeline 
This is an NLP (Natural Language Processing) pipeline project that focuses on sentiment analysis using the IMDb dataset. The project performs the following steps:

## 1. Load dataset
The IMDb dataset is loaded from the 'IMDB.csv' file.

## 2. Data Cleaning
The dataset reviews are cleaned using the following steps:

Remove HTML tags from the reviews.
Remove special characters from the reviews.
Convert all text to lowercase.
Remove stopwords from the reviews.
Perform stemming on the words in the reviews.

## 3. Model Creation
The project creates a sentiment analysis model using the following steps:

Creating a Bag of Words (BOW) representation of the cleaned reviews.
Splitting the data into training and testing sets.
Defining three Naive Bayes models (GaussianNB, MultinomialNB, and BernoulliNB).
Training the models on the training data.
Saving the trained models to disk.
Making predictions using the trained models on the testing data.

## 4.Model Evaluation
The project evaluates the performance of the models by calculating the accuracy of the predictions on the testing data.






Intel DevMesh Link - https://devmesh.intel.com/projects/movie-reviews-sentiment-analysis

<!-- PREREQUISTIES -->
## <br>**➲ Prerequisites**
This is list of required packages and modules for the project to be installed :
* <a href="https://www.python.org/downloads/" target="_blank">Python 3.x</a>
* Pandas 
* Numpy
* re
* Scikit-learn
* NLTK

## <br>**➲ The Dataset**
Human activites dataset contain about 5000 record which is a sample of movie's review<br>
and a target column "sentiment" which describe the sentiment of the viewer about the movie either it is positove or negative<br>
<br>**Dataset features and target :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/3933cc5e-d891-4fc1-a3c0-637402c7ee12)
<br>


<!-- CODING SECTIONS -->
## <br>**➲ Coding Sections**
In this part we will see the project code divided to sections as follows:
<br>

- Section 1 | Data Preprocessing :<br>
In this section we aim to do some operations on the dataset before training the model on it,
<br>processes like :
  - Loading the dataset
  - Encoding ouput to binary (Positive : 1 , Negative : 0) 
  - Data cleaning : Remove HTML tags
  - Data cleaning : Remove special characters
  - Data cleaning : Convert everything to lowercase
  - Data cleaning : Remove stopwords
  - Data cleaning : Stemming<br><br>

- Section 2 | Model Creation :<br>
The dataset is ready for training, so we create a Naive Bayes model using scikit-learn and then fit it to the data.<br>

- Section 3 | Model Evaluation :<br>
Finally we evaluate the model by getting accuracy, classification report and confusion matrix.

<!-- INSTALLATION -->
## <br>**➲ Installation**
1. Clone the repo
   ```sh
   git clone https://github.com/omaarelsherif/Movie-Reviews-Sentiment-Analysis-Using-Machine-Learning.git
   ```

<!-- OUTPUT -->
## <br>**➲ Output**
Now let's see the project output after running the code :

**Dataset head :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/3933cc5e-d891-4fc1-a3c0-637402c7ee12)

**Dataset after output encoding :
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/cc6d8e42-7fb6-48f8-a5d4-2108fe10f1b4)

**Review sample after removing HTML tags :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/d2163ca0-2697-4c9b-85b6-cc627fc24220)

**Review sample after removing special characters :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/810a7505-ce63-4169-8677-fef30f318535)

**Review sample after converting words to lowercase :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/229b29fd-6283-4dd7-a3bf-21f9a7ab8606)

**Review sample after removing stopwords :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/51e1c087-58e2-4ea5-8b09-3aee63ba2a5c)

**Review sample after stemming words :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/fd676ee8-eb9d-405e-94c3-b4c1c817fdb3)


**Bag Of Words "BOW" and Models accuracy :**<br>
![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/cb23540c-d3d2-4ab6-a7fd-8efae9c67a17)

![image](https://github.com/viveklistenus/Sentiment_Analysis_using_IMDB_Reviews/assets/28853520/34bb1784-c9e8-4055-a1fc-079793ac72c6)





<!-- REFERENCES -->
## <br>**➲ References**
These links may help you to better understanding of the project idea and techniques used :
1. Natural Language Processing (NLP) : https://ibm.co/38bN03T
2. Sentiment analysis : https://bit.ly/3yi9BGq
3. Naive Bayes classifier : https://bit.ly/3zhoWIO
4. Model evaluation : https://bit.ly/3B12VOO
