# **DEVELOPING A PREDICTIVE MODEL FOR EARLY DETECTION OF MENTAL HEALTH CONDITIONS** 

**Authors:**

Elvis Wanjohi (Team Leader)

Jessica Gichimu

Jesse Ngugi

Stephen Gachingu

Latifa Riziki

## 1. Business Understanding

### 1.1 Business Overview
Given the fast-moving pace of economic and technological advancement in today’s world, most people, especially from the younger generation, tend to experience some form of mental health issues in their lifetime. There has been a significant increase in individuals experiencing suicidal ideation. While it may appear that such individuals do not explicitly communicate their distress, a closer examination of their online activity, such as social media posts, comments, and engagement patterns, often reveals underlying emotional states indicative of psychological distress. This could help researchers, students, and practitioners to develop early detection models for mental health support. The goal is to encourage data-driven approaches to mental health awareness, prevention, and support systems. Mental health awareness is primarily in the healthcare and psychology domains, focusing on the assessment, diagnosis, and treatment of mental health conditions. 

The target audience for this NLP model are health care professionals (such as therapists, psychologists, psychiatrists), and mental health organizations and clinics, where they can prioritize high-risk cases, or monitor trends in mental health conditions across populations. This model could be used to identify early symptoms of the mental health of individuals in our society. We were able to find a brief description of mental health in the Practical Natural Language Processing( A Comprehensive Guide to Building Real-World NLP Systems) book, which gave us the idea of tackling this project. The motivation for the project is try and improve the diagnosis  and treatment of mental health by identifying underlying conditions at an early stage.
### 1.2 The Problem
Mental health professionals often rely on personal expertise and manual assessment to diagnose patients—an approach that is time-intensive and difficult to scale. In many cases, warning signs are missed or detected late, especially when individuals express themselves informally online.

This project proposes an NLP-based predictive model that analyzes text statements and classifies potential mental-health conditions. By integrating automated linguistic analysis with professional oversight, the model can complement human judgment, enabling faster, broader screening. The tool is also designed for public awareness and self-assessment purposes, with a clear ethical guideline that professional consultation must accompany any automated insights.

### 1.3 Project Objectives
#### 1.3.1 Main Objective
The main objective is to develop a machine learning model that can accurately classify mental health conditions based on textual statements expressed by individuals.

#### 1.3.2 Specific Objectives
The specific objectives of the project are:
 
1. Translate all text data into Swahili to localize the dataset and improve inclusivity.

2. Identify the most common mental health condition.

3. Preprocess the data through processes such as Vectorization and tokenization, handling missing values, and creating new features such as characters, words and sentences.

4. Use exploratory tools such as word clouds to visualize commonly terms associated with specific mental health categories.

5. Analyze text length to classify a mental health condition or show correlation  with a mental health condition. 

6. Evaluate model performance using metrics such as Precision, Recall, F1score, Accuracy Score and ROC-AUC.

7. Compare different classification models to determine which performs best for this dataset.

8. Scrapping data from an online platform like twitter to show the efficiency of the model.

9. Create a translate feature to allow English–Swahili switching for interpretability and diversity in the model.


#### 1.3.3 Research Questions
1. Can the dataset be effectively translated and localized to Swahili?

2. Which is the most common health condition?

3. Which features influence mental health condition?

4. Which words are specific to each mental health category?

5. Which classifier model achieves the best Precision, Recall, F1 score, Accuracy and ROC-AUC?

6. Which classification model performs best for this dataset?

7. How efficiently can the model classify conditions when applied to Twitter data?

8. How can we ensure diversity, fairness and interpretability in the multilingual model?


### 1.4 Success Criteria
The success of this project will be assessed in the following ways:

1. The analysis should generate actionable insights into the most common mental health conditions to inform better prevention and support strategies.

2. A machine learning model should successfully classify text into relevant mental health categories with high performance metrics.

3. The final system should maintain interpretability, cultural sensitivity and ethical integrity when applied to both English and Swahili datasets.

## 2. Data Understanding
This section makes use of the English mental health text dataset to build a mental health condition classification model. We also translated the dataset into swahili to show the disparity between English and Swahili text. The project also aimed to use statements in swahili language, serving as the foundation for downstream Natural Language Processing (NLP) modeling in an African context.

The aim is to understand the dataset’s structure and content. This includes reviewing the available features, verifying data types and identifying potential quality issues such as missing values, duplicates or inconsistencies.

By exploring the data at this stage, it is possible to detect quality concerns and inform decisions for text cleaning, data preprocessing and subsequent model development.

## 3. Data Preparation
The aim of this step is to transform raw, unstructured text into a clean, structured, and meaningful format that machine learning models can understand and learn from effectively. This involves steps such as;
- Removing noise (like punctuation, URLs, and stopwords)
- Normalizing text through techniques like tokenization, stemming, or lemmatization
- Pos tagging- It is short for part of speech and involves assigning each text a grammatical category like noun, verb, and adjective.
- Converting words into numerical representations through TF-IDF.
  
Proper data processing ensures that linguistic patterns are preserved while irrelevant information is minimized, improving model accuracy, reducing bias, and enhancing the efficiency and interpretability of NLP systems.

## 4. Exploratory Data Analysis
The dataset was explored to understand patterns across user text and the different mental health classes . Key analysis included:

- Mental Health Label Distribution Analysis- This was done to check for class imbalance in the dataset.
- Text Length Distribution Analysis- This section examines the distribution of text lengths in the dataset to understand variation across both English and Swahili entries. The analysis helps identify whether most posts are short or long, which influences preprocessing and model selection decisions.
- Text Length by Mental Health Label- This section examines how the word count of posts varies across different mental health labels.
- Distribution of Text length/Character, Word and Sentence Counts- This section examines the basic linguistic structure of the dataset by analyzing the distribution of character, word, and sentence counts across both the English and Swahili text columns. Understanding these linguistic features helps reveal how post length and complexity vary between languages and across the different mental_health_label categories.

## 5. Modeling and Evaluation

For this section, traditional models and advanced deep learning models, RoBERTa( Robustly Optimized BERT Pretraining Approach) were developed and evaluated. They included;

1. **Logistic Regression**: A linear model used for binary and multiclass classification.

2. **Naive Bayes**: A probabilistic classifier commonly applied in text analysis.

3. **LinearSVC**: A support vector machine variant optimized for high-dimensional text data.

4. **Random Forest**: An ensemble model that combines multiple decision trees for robust predictions.

### 5.1 Multi-Class Classification
The steps for multi-class modeling are as follows:

- Defining X and Y Features.
- Encoding the labels(Positive, Negative and Neutral)
- Split the data.
- Train the models with each model having a baseline model and then do hyperparamater tuning to find the best model.
- Create a pipeline that trains and tests our data. The Pipeline contains TfidfVectorizer which is a vectorizer that converts our textual data to numeric data.
- Evaluate the models using training accuracy,validation accuracy, F1 score and ROC.
- Plotting the Accuracy score, F1 score, ROC and Confusion matrix.
- Save the best performing model.

**Key Observations:**
* The best performing traditional models are Logistic Regression and Linearsvc with LinearSVC being the slightly better model.
* The LinearSVC model had  the best  Validation accuracy of **87.3%**. The Logistic regression model followed closely behind with a validation accuracy  of **86.6%**.
* The weakest model was the Random Forest model with a validation accuracy of **76.2%** but the model is stable since it still has a slight drop from training to validation accuracy.

### 5.2 Deep Learning
For the deep learning approach, the **Roberta model** (short for Robustly Optimized BERT Pretraining Approach) which is a transformer model, was used. It's an optimized version of **BERT** that is designed to understand and interpret human language in a way that is more accurate and efficient. The steps were as follows;

1. Importing necessary dependencies
2. Loading the dataset and cleaning
3. Turning text data into numbers(label encoding)
4. Splitting the dataset into train, validation and test.
5. Loading the Roberta tokenizer.
6. Tokenizing the text data.
7. Loading the Roberta model for Classification.
8. Defining how we measure success using Accuracy score.
9. Setting up training instructions.
10. Building the trainer.
11. Training the model.
12. Visualizing the training, validation accuracy, and loss.
13. Save the model.
    
**Key Observations:**
* After running the model for 3 epochs we have a final accuracy of 88.59%, this  makes it the best model compared to the    traditional models we trained earlier
* The Training loss reduces with each epoch meaning the model was learning well and generalizing properly which is further emphasized by the minimal gaps between each training loss
* There's minimal increase in the  validation loss from  (0.35 → 0.41). This suggests minimal overfitting in our deep learning model as  it learns training data better than new data.

* Overall, the model performance is strong and stable, nearly 88–89% accuracy

## 6.Webscrapping

In this section we fetched tweets from twitter to check the efficiency of the best traditional model. 


## 7. Deployment
The trained model was deployed to Streamlit Cloud to make it interactive and easy to use.
Key Steps:
1. Prepared a requirements.txt file listing all required Python packages.
2. Create an app.py that contains all the functionalites which include loading the saved model and e.t.c
3. Upload the heavy models to hugging face repository, due to challenges of uploading on Github.
4. Push the code to Github.
5. Deploy the app on streamlit.

App Link:
https://globalmentalhealthapp-niqkscywvz7gh3ryy65hyy.streamlit.app/


## 8. Technologies Used

- **Python**: Primary programming language

- **Pandas**: Data manipulation and analysis

- **Sklearn**: For modelling purposes

- **Matplotlib**: Data visualization

- **Jupyter Notebook**: Development environment

- **Git and Github**: Version control and remote repository management

- **NLTK**: For Natural Language Processing

- **Plotly**: To create interactive visualizations

- **Joblib and Pickle**: To save the trained model.
  
- **Transformers (Hugging Face)**: For implementing the Roberta deep learning model

- **PyTorch**: Deep learning framework used to train transformer-based models

- **Wordcloud**: This was used to get the most recurring words in the different mental health conditions.

- **Streamlit**: For deploying the model

- **Tweepy**: This was used for webscrapping


## 9. Conclusion

This project explored how Natural Language Processing (NLP) and machine learning can detect mental health conditions from text, including a Swahili-translated dataset to make the model more inclusive. Traditional models like Logistic Regression and LinearSVC performed well, but the deep learning approach using the Roberta model (Robustly Optimized BERT Pretraining Approach) achieved the best results. Roberta, a transformer-based model that improves on BERT, was trained through steps such as data cleaning, tokenization, model loading, training, and evaluation. After 3 epochs, it reached an accuracy of 88.59%, showing strong learning ability with steadily decreasing training loss and only a small increase in validation loss (0.35 to 0.41), suggesting slight overfitting. Overall, the Roberta model demonstrated high accuracy and effectiveness in understanding language and detecting patterns in mental health data, outperforming traditional machine learning models.

## 10. Support
For questions or support, please contact:

1. jessengugi99@gmail.com

2. jessica.gichimu@gmail.com

3. stephenmunene092@gmail.com

4. leeelvis562@gmail.com

5. latifariziki5@gmail.com

