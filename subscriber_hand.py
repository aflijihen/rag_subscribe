import paho.mqtt.client as mqtt
import json
import time
import os
from openai import OpenAI
from datahandler import DataHandler
from datahandler import DataHandler 
from pushbullet import Pushbullet
import requests
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
import docx2txt





# Charger les variables d'environnement
load_dotenv()

# Variables globales
broker_address = "mqtt.eclipseprojects.io"
topic = "Spirulina_Edge"
API_KEY = os.getenv('PUSHBULLET_API_KEY')
pb = Pushbullet(API_KEY)

class Subscriber:
    def __init__(self, data_handler, llm_model):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.data_handler = data_handler
        self.llm_model = llm_model
        self.processing_message = False 
        
    def start(self):
        self.client.connect(broker_address)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to MQTT broker with result code " + str(rc))
        self.client.subscribe(topic)

    def on_message(self, client, userdata, msg):
        if not self.processing_message:
            self.processing_message = True 
            data = json.loads(msg.payload.decode("utf-8"))
            json_data = json.loads(data)
            print("Received data:", json_data)
           
            self.data_handler.execute(json_data)
            self.processing_message = False

   
# Exemple d'utilisation
if __name__ == "__main__":
    data_handler_instance = DataHandler()
    llm_model_instance = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    subscriber = Subscriber(data_handler=data_handler_instance, llm_model=llm_model_instance)
    subscriber.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping subscriber...")
        subscriber.stop()
        print("Subscriber stopped.")
