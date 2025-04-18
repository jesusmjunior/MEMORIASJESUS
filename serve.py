from flask import Flask, render_template

app = Flask(__name__)  # Usa a pasta padrão 'templates'

@app.route("/")
def home():
    return render_template("memory.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
