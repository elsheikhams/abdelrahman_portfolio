## Project #1 | Delirium Classification Model

Developed a machine learning classification model that predicts whether a patient has delirium or not. The model achieved an overall accuracy score of 96% on the test set.

You can view the source file at [Delirium Classification](https://github.com/elsheikhams/classification-1)

- Performed Data Cleaning and Preprocessing
- Model Hyperparameter Tuning
- Model Training and Evaluation

```
# Splitting the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y)

# Oversampling
X_train, y_train = ADASYN(random_state=0).fit_resample(X_train, y_train)

# Applying feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Training and predicting
classifier = SGDClassifier(alpha= 0.0001, early_stopping= True, loss= 'hinge', max_iter= 1000, penalty= 'l1', random_state= 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Testing the accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy ' + str(accuracy_score(y_test, y_pred)))
```
[[39  2]<br>
[ 0 11]]<br>
Accuracy 0.9615384615384616

## Project #2 | Company Sector Classification (Text Classification)

Developed a machine learning classification model that predicts what the company sector is given the textual description of the company. The model achieved an overall accuracy score of 91.46% on the test set.

You can view the source file at [Company Sector Classification](https://github.com/elsheikhams/company_sector_classification)

- Performed Textual Data Cleaning
- Natural Language Processing (NLP)
- Model Hyperparameter Tuning
- Model Training and Evaluation

```
classifier = LinearSVC(fit_intercept= True, loss= 'hinge', max_iter= 900, multi_class= 'crammer_singer', random_state= 42)
classifier.fit(X_traincv, y_train)
y_pred = classifier.predict(X_testcv)

# Testing the accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy ' + str(accuracy_score(y_test, y_pred)))
```
Accuracy 0.9146285186894324


## Project #3 | Arabic Dialect Classification (Text Classification)

A multi-class text classification model that predicts Arabic dialects given a text. Both machine learning models and deep learning architectures were used in this project. The highest accuarcy was achieved by using LinearSVC() classifier in the case of the machine learning models. The deep learning model, on the other hand, achieved less validation accuarcy. The deep learning architecture used in this classification problem was LSTM, with 3 epochs.

You can view the source file at [Arabic Dialect Classification](https://github.com/elsheikhams/arabic_text_classification)

- Data Retrieval from API
- Performed Textual Data Cleaning
- Natural Language Processing (NLP)
- Model Hyperparameter Tuning
- Model Training and Evaluation
- Deep Learning Model
- Model Deployment
