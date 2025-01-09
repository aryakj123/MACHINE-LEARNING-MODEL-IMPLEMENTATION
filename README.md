**Name:** ARYA K J

**Company:** CODTECH IT SOLUTIONS

**ID:** CT08EQI

**Domain:** Python Programming

**Duration:** December 20,2024-January 20,2025

**Mentor:** SRAVANI GOUNI


### Description of Task ###
This project demonstrates the implementation of an SMS Spam Detection system using Python and Machine Learning techniques. The system is built using the Logistic Regression algorithm and leverages TF-IDF Vectorization for text preprocessing. It serves as a complete example of text classification, from data preparation to model evaluation and visualization.

Purpose of the Project
The goal of this project is to classify SMS messages into two categories:

Ham (Not Spam)
Spam (Unwanted or promotional messages)
This classification helps in filtering out spam messages, improving user experience, and ensuring privacy and security in communication.

Code Overview
1. Data Loading and Preparation
The dataset used is the publicly available SMS Spam Collection Dataset. The code performs the following:

Loads the dataset from a remote CSV file.
Drops unnecessary columns to focus on relevant data.
Renames columns for clarity:
v1 → label (Ham or Spam)
v2 → text (SMS content)
Converts labels to numerical format for machine learning:
0 for Ham
1 for Spam
This ensures the data is clean and ready for further processing.

2. Exploratory Data Analysis (EDA)
Data Type Check: Verifies the structure and types of the dataset to ensure consistency.
Missing Values: Checks for and confirms there are no missing values.
3. Text Preprocessing with TF-IDF
TF-IDF Vectorization: Converts raw text into numerical features that the model can understand.
Removes stop words (common words like "the," "is") to focus on meaningful content.
Normalizes the data by considering the importance of each word in relation to the entire dataset.
The training and test sets are vectorized separately to avoid data leakage.
4. Model Training: Logistic Regression
A Logistic Regression model is chosen for its simplicity and effectiveness in binary classification tasks like spam detection.
The model is trained on the vectorized training set to learn patterns distinguishing ham and spam messages.
5. Model Evaluation
The trained model is evaluated using multiple metrics to assess its performance:

Accuracy: Measures the overall correctness of predictions.
Classification Report: Provides detailed metrics like Precision, Recall, and F1-Score for each class.
Confusion Matrix: Summarizes the prediction performance with:
True Positives (Correct Spam Predictions)
True Negatives (Correct Ham Predictions)
False Positives (Ham misclassified as Spam)
False Negatives (Spam misclassified as Ham)
6. Visualization
A heatmap of the confusion matrix is created using Seaborn, providing a visual understanding of the model’s performance. The heatmap makes it easier to identify where the model is making errors.

7. Example Predictions
To demonstrate the model's real-world application, new SMS messages are manually tested. The model predicts whether each message is spam or ham, showcasing its ability to generalize to unseen data.

Example Messages:

Spam: "Congratulations! You've won a free vacation!"
Ham: "Hi, how are you doing today?"
Why This Project is Important
Practical Application: Spam filtering is a crucial feature in modern communication platforms.
Machine Learning Concepts: Provides a hands-on understanding of:
Text preprocessing with TF-IDF
Binary classification with Logistic Regression
Model evaluation and visualization
Scalable Solution: The pipeline can be extended to larger datasets or more complex models (e.g., Naive Bayes, Support Vector Machines, or Neural Networks).
Project Structure
spam_detection.py: The main Python script implementing the spam detection system.
requirements.txt: Contains all the dependencies needed to run the project.
Dataset Source: The SMS Spam Collection Dataset is fetched directly from the raw URL for seamless integration.
How to Run
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Run the script:

bash
Copy code
python spam_detection.py
The script will:

Train the model.
Evaluate its performance.
Display visualizations.
Predict outcomes for example SMS messages.
Future Enhancements
Advanced Models: Implement additional models like Naive Bayes or Neural Networks for improved performance.
Real-Time Predictions: Build a web or mobile interface for real-time spam classification.
Dataset Expansion: Incorporate more diverse datasets for better generalization.
This project is a great starting point for anyone looking to explore text classification or build practical machine learning systems. It’s simple, effective, and easy to extend for more complex use cases.






