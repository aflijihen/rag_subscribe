import paho.mqtt.client as mqtt
import json
from openai import OpenAI
import os  
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()



broker_address = "mqtt.eclipseprojects.io"
topic = "Spirulina"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Publisher:
   
    def __init__(self) :
         self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
     
    
    def generate_recommendation(self, data):
        # Construct user input string summarizing sensor data
        user_input=f"Données traitées :\n\n" \
                     f"- Température : {data['temperature']}°C\n" \
                     f"- pH : {data['Ph_value']}\n" \
                     f"- Niveau d'eau : {data['water_level']} mètres\n" \
                     f"- Conductivité : {data['conductivity']} µS/cm\n" \
                     f"- Luminosité : {data['brightness']} lux\n\n" \
                     f"Utiliser ces données pour générer des recommandations et les publier."

        template_prompt = """
Assurez-vous que toutes les réponses sont au format JSON et suivent la structure suivante :

Exemple de réponse JSON contenant les codes de commande :
{
    "ph_prediction": 0,
    "ec_prediction": 1,
    "temp_prediction": 3
}

Assurez-vous que chaque code de commande correspond correctement à son intention et à son utilisation prévue dans votre système.

Veuillez fournir les réponses sous forme JSON avec la clé "command"  représentant les prédictions ou les commandes pour chaque mesure.

Exemple d'utilisation :

{
    "Commands": {
        "ph_prediction": 2,
        "ec_prediction": 0,
        "temp_prediction": 1
    }
}

Merci d'utiliser ce format pour faciliter le traitement automatique des données.
"""

        # Ajout de la prompt à l'input utilisateur
        user_input_with_prompt = f"{template_prompt}\n\n{user_input}"
        # Generate response using the=
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": template_prompt},
                {"role": "user", "content": user_input_with_prompt},
            ],
        )
        return response.choices[0].message.content
 
        #return (response.choices[0].message.content )# Convert to uppercase for easier matching


    def publish_recommendations(self, recommendations):
        
        # Connect to MQTT broker
        client = mqtt.Client()
        client.connect(broker_address)
         # Convertir les recommandations extraites en JSON et publier au topic
        message = json.dumps(recommendations)
        client.publish(topic, message)
        print("Published recommendations:", recommendations)



  

