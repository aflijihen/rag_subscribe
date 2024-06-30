# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
# import paho.mqtt.client as mqtt
# import json
# from openai import OpenAI

# # Charger les variables d'environnement depuis le fichier .env
# load_dotenv()

# class DataHandler:
#     def __init__(self):
#         self.docs_dir = "./handbook/"
#         self.persist_dir = "./handbook_faiss"
#         self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
#         self.build_faiss_index()
    
#     def build_faiss_index(self):
#         # Construction de l'index FAISS à partir des documents
#         loader = DirectoryLoader(
#             self.docs_dir,
#             loader_cls=Docx2txtLoader,
#             recursive=True,
#             silent_errors=True,
#             show_progress=True,
#             glob="**/*.docx"
#         )
#         # À compléter en fonction de votre logique spécifique

#     def execute(self, data):
#         # Extraction des valeurs des données reçues
#         temperature = {data["temperature"]}
#         ph_value ={ data["Ph_value"]}
#         water_level ={ data["water_level"]}
#         conductivity ={ data["conductivity"]}
#         brightness ={ data["brightness"]}

#         # Génération du prompt basé sur les données reçues
#         prompt = f"""
# Basé sur les valeurs suivantes :
# - Température : {temperature}°C
# - pH : {ph_value}
# - Niveau d'eau : {water_level} mètres
# - Conductivité : {conductivity} µS/cm
# - Luminosité : {brightness} lux

# Générer des recommandations et des commandes au format suivant à partir des documents fournis :

# {{
#     "Recommendations": {{
#         "pH": "Ajouter de la soude caustique (NaOH).",
#         "Conductivity": "Vider 20% du bassin et le remplacer par de l'eau fraîche.",
#         "Température": "Plage idéale."
#     }},
#     "Commands": {{
#         "ph_prediction": 2,
#         "ec_prediction": 0,
#         "temp_prediction": 1
#     }}
# }}

# Exemple de recommandations :

# Température : 25°C, pH : 7.5, Niveau d'eau : 1.5 mètres, Conductivité : 500 µS/cm, Luminosité : 2000 lux

# {{
#     "Recommendations": {{
#         "pH": "Plage idéale.",
#         "Conductivity": "Plage idéale.",
#         "Température": "Plage idéale."
#     }},
#     "Commands": {{
#         "ph_prediction": 1,
#         "ec_prediction": 1,
#         "temp_prediction": 1
#     }}
# }}
# """

#         # Utilisation de l'API OpenAI pour obtenir des recommandations
#         response = self.llm.chat.completions.create(
#             model="gpt-3.5-turbo",
#             max_tokens=500,
#             temperature=0.7,
#             messages=[
#                 {"role": "system", "content": "Vous êtes un assistant de recommandation."},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#         return response.choices[0].message.content
import os
import json
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
import docx2txt
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema.output_parser import StrOutputParser
from llama_index_client import PgVectorStore
from langchain_openai import ChatOpenAI
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from publisher_handler import Publisher
import http.client, urllib
from pushbullet import Pushbullet

load_dotenv()
class DataHandler:
    def __init__(self):
        self.docs_dir = "./handbook/"
        self.persist_dir = "./handbook_faiss"
        self.embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.6)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.API_KEY = os.getenv('PUSHBULLET_API_KEY')
        
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
      
        
        self.load_or_build_faiss()
        self.initialize_qa_chain()
        self.build_faiss_index()
        
    def load_or_build_faiss(self):
        if os.path.exists(self.persist_dir):
            print(f"Loading FAISS index from {self.persist_dir}")
            self.vectorstore = FAISS.load_local(self.persist_dir, self.embedding, allow_dangerous_deserialization=True)
            print("Done.")
        else:
            print(f"Building FAISS index from documents in {self.docs_dir}")
            pass

    def initialize_qa_chain(self):
        #utilisée pour initialiser une chaîne de traitement de récupération conversationnelle 
        self.qa_chain = ConversationalRetrievalChain.from_llm(
             llm=self.llm,
             memory=self.memory,
             retriever=self.vectorstore.as_retriever()
   
        )
    def build_faiss_index(self):
        # Construction de l'index FAISS à partir des documents
        loader = DirectoryLoader(
            self.docs_dir,
            loader_cls=Docx2txtLoader,
            recursive=True,
            silent_errors=True,
            show_progress=True,
            glob="**/*.docx"
        )
   
  
      
      
    def execute(self, data):
            
        self.API_KEY = os.getenv('PUSHBULLET_API_KEY')
       
        
        user_input = (f"Utiliser les valeurs actuelles : temperature: {data['temperature']} - Ph : {data['Ph_value']} - Niveau_deau : {data['water_level']} - Conductivité : {data['conductivity']} - Luminosité : {data['brightness']}, donner des recommandations spécifiques à chaque mesure : ")
            # Ajout d'une prompt pour guider la génération des recommandations
        sys_prompt = (
    f"Pour chaque mesure, veuillez fournir une recommandation et des commandes à partir de document au format ci-dessous :\n\n"
    f"Recommendations:\n"
    f"1. Température ({data['temperature']}°C) :\n"
    f"   - Recommandation : *idéale ou non idéale*, suivi d'une explication détaillée.\n"
    f"2. pH ({data['Ph_value']}) :\n"
    f"   - Recommandation : *idéale ou non idéale*, suivi d'une explication détaillée.\n"
    f"3. Niveau d'eau ({data['water_level']} mètres) :\n"
    f"   - Recommandation : *idéale ou non idéale*, suivi d'une explication détaillée.\n"
    f"4. Conductivité ({data['conductivity']} µS/cm) :\n"
    f"   - Recommandation : *idéale ou non idéale*, suivi d'une explication détaillée.\n"
    f"5. Luminosité ({data['brightness']} lux) :\n"
    f"   - Recommandation : *idéale ou non idéale*, suivi d'une explication détaillée.\n\n"
   
)
  


        

            
        if user_input.lower() == "exit":
                return
        else:
                result = self.qa_chain.invoke({"question": user_input})
                generated_response = result["answer"]
                print("Recommendations:", generated_response)
                
                
                # Extract codes from recommendations
                publisher = Publisher()
                recommandations = publisher.generate_recommendation(data)
                publisher.publish_recommendations(recommandations)
        
           
    
               
        
               

                
                    
                
                    
                


