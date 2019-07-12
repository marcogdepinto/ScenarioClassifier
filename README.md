# Scenario Classifier

Classifying different scenarios (city, desert, mountain, nature, sea, universe).

Project built using Python: Keras (with Tensorflow backend), numpy and scikit-learn.

**Dataset structure**

The dataset is made of six folders (city, desert, mountain, nature, sea, universe), each one containing 25 pictures. An example of city picture is provided below.

![Link to example picture](https://github.com/marcogdepinto/ScenarioClassifier/blob/master/examples/2.GettyImages-187703420.jpg)

**How to install it**

```$ git clone https://github.com/marcogdepinto/ScenarioClassifier.git```

```$ pip install -r path/to/requirements.txt```

**How to run it**

1) Download the model from [this link](https://drive.google.com/open?id=1jPDQcqQeh7r-_yQgn9jJzc8yVXEOgkvk) (it exceeds 100 megabytes so it can't stay on Github) and place it in a folder called ```model``` in the main directory of the project.

2) Run ```predictions.py```. The script will return the classification report and the confusion matrix of the model. 

**Achievements**

Actually the model has an **F1 score of 91% on test data**.

If you want to train again the model changing the parameters, do your changes in ```train.py``` and then launch it.

**Metrics (Classification Report and Confusion Matrix)**

![Link to metrics](https://github.com/marcogdepinto/ScenarioClassifier/blob/master/metrics/metrics.png)

**Model structure**

![Link to model structure](https://github.com/marcogdepinto/ScenarioClassifier/blob/master/model.png)
