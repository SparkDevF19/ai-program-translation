from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer #Turns text comments into vectors
from sklearn.naive_bayes import MultinomialNB #Module Classifier
from sklearn.externals import joblib #SQL?
 
app = Flask(__name__)

#URL location, home page "/" empty slash
@app.route('/')
def home():
	return render_template('home.html')

# Url location... .com"/predict"
@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("YoutubeSpamMergedData.csv")
	df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	df_x = df_data['CONTENT']
	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	# Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
    # Fit training
	clf.fit(X_train,y_train)
    # Check accuracy, test
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	## ytb_model = open("naivebayes_spam_model.pkl","rb")
    # Machine learning part, loads saved model
	## clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
        # Vectorizes comments
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)


# Link for tut
# 	https://www.youtube.com/watch?v=tFjeUtFay_Q&t=966s
