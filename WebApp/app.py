
# How to run the Flask Web APP on your machine
# Every folder necessary to run the virtual enviornment should be already there for you
# cd to WebApp
# env\Scripts\activate
# flask run then ctrl+click the link and there it is!

import json
from flask import Flask, render_template, request

app = Flask(__name__,
    static_url_path='',
    static_folder='C://Users/Chelsea/Desktop/ai-program-translation/WebApp/static',
    template_folder='C://Users/Chelsea/Desktop/ai-program-translation/WebApp/templates')

#URL location, home page "/" empty slash
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/Page2')
def page2():
    return render_template("page2.html")

@app.route('/TranslationPage')
def translation():

    text = open("text.txt", "r")
    read = text.read()
    split = read.split ("asdf")
    placeholderT = str(split[0])

    return render_template("translation.html", placeholderT=placeholderT)


@app.route('/translation', methods = ['POST'])
def translation_post():
    if request.method == 'POST':
        inputCode = request.form['inputCode'] 
        processed_text = inputCode.upper()
        
        return render_template("translation.html",  processed_text)

@app.route('/training')
def training():
        return render_template("training.html")



if __name__ == '__main__':
    app.run(debug=True)
        
	
