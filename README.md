# Our project will talk about AI in Cybersecurity.

# We will include also the source code of the phishing detection program


## How to use the program? 

First you should have some libraries already such as ``flask, scikit-learn, numpy, joblib``

to install it run this command line (make sure you have python 3.8 or a newer version.)

```pip3 install flask scikit-learn numpy joblib```

<b>what are those libraries used for?</b>
scikit-learn - Machine learning<br>
numpy - Numerical Processing<br>
joblib - Saving/loading the model<br>
flask - running a web interface

## Training the model

```python3 Phishing_Det.py train --model-out phishing_model.joblib --cv 5```

Ok what are those parameters we used? 

``traing`` --> starts the training process
``--model-out`` --> File name where the trained model will be saved
`` --cv 5`` --> this one is optional, but recommended. (since i will upload the trained model file it's not necessary to run this command line.)

## Running the program

Now you can run it by using this command
```python3 Phishing_Det.py predict --model phishing_model.joblib --interactive```

<b>How it will work?</b>
The program will ask you to enter the message you recieved
After that you will pick the source, email / sms / unknown.
and lastly you will enter the sender domain name, or the organization that sent you

The AI will then analyze the message and output
Prediction (Phishing or Safe)
Confidence score
Reason for the decision.

# Running the program using web interface (Recommended) 

```python3 app.py```
Simple as that, it will run locally.

