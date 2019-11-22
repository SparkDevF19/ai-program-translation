from flask import Flask, render_template, request

app = Flask(__name__)

#URL location, home page "/" empty slash
@app.route('/')
def home():
    return render_template("translation.html")


@app.route('/translation', methods = ['POST'])
def translation():
    if request.method == 'POST':
        comment = request.form['code']
        
        code = comment
        code = code.upper()
        return render_template("translation.html", comment=comment, code = code)


if __name__ == '__main__':
	app.run(debug=True)
