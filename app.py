from flask import Flask, render_template, request, jsonify
import json
from chat import get_response
import os

app = Flask(__name__)

# Définir le chemin du fichier pour stocker les conversations
conversation_file = "conversations.json"

# Fonction pour enregistrer la conversation dans un fichier JSON
def save_conversation(user_message, bot_response):
    # Vérifiez si le fichier existe déjà, sinon, créez une nouvelle liste
    if os.path.exists(conversation_file):
        with open(conversation_file, "r") as f:
            conversations = json.load(f)
    else:
        conversations = []

    # Ajoutez le nouvel échange à la liste
    conversations.append({"user": user_message, "bot": bot_response})

    # Enregistrez les conversations mises à jour dans le fichier
    with open(conversation_file, "w") as f:
        json.dump(conversations, f, indent=4)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # vérifier si le texte est valide 
    response = get_response(text)

    # Enregistrer la conversation
    save_conversation(text, response)

    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
