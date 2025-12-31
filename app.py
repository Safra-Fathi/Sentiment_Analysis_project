from flask import Flask, render_template, request, redirect
from predictionpipeline import preprocessing, vectorizer, get_prediction

app = Flask(__name__)

reviews = []
positive = 0
negative = 0

@app.route("/", methods=["GET", "POST"])
def index():
    global positive, negative

    if request.method == "POST":
        text = request.form["text"]

        preprocessed_txt = preprocessing(text)
        vectorized_txt = vectorizer(preprocessed_txt)
        prediction = get_prediction(vectorized_txt)

        if prediction == "negative":
            negative += 1
        else:
            positive += 1

        reviews.insert(0, text)
        return redirect("/")

    data = {
        "reviews": reviews,
        "positive": positive,
        "negative": negative
    }

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
