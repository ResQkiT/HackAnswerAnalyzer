from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Данные, которые передаем в шаблон
    user = {"name": "Алиса"}
    items = ["Яблоко", "Банан", "Апельсин"]
    return render_template('index.html', user=user, items=items)

if __name__ == '__main__':
    app.run(debug=True)