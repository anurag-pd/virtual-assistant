#import all the libraries required
import csv, pickle, numpy as np, os
from sentence_transformers import SentenceTransformer, util
#Virtual Agent Model
model= SentenceTransformer("stsb-mpnet-base-v2") #load pretrained model
qa = {}
emb = []
    #train virtual assistant
def train(training_file):
    global qa,emb
    # if model doesn't exist in the location, compute embeddings again and store as a model
    if not os.path.exists(r"models/model_va.pickle"):
        header = False
        dict_model = dict()
        with open(training_file, "r", encoding="utf-8", errors="ignore") as file:
            reader = csv.reader(file)
            for qa_pair in reader:
                qa[qa_pair[0]] = qa_pair[1]
                emb.append( model.encode(qa_pair[0])) #compute embeddings
            dict_model["qa"] = qa
            dict_model["embeddings"] = emb
        #persist trained model
        with open(r"models/model_va.pickle", "wb") as file:
            pickle.dump(dict_model, file)
#predict answer to user query
def pred_answer(usr_query):
    global qa,emb
    query_embedding = model.encode(usr_query) #compute embedding for the user query
    if not qa and not emb: #load trained model if not done already
        with open(r"models/model_va.pickle", "rb") as file:
            dict_model = pickle.load(file)
            qa = dict_model["qa"]
            emb = dict_model["embeddings"]
    sim_scores = util.pytorch_cos_sim(query_embedding,  emb) #computet similarity scores
    matched_query = list( qa.keys())[np.argmax(sim_scores)] #identify matched query based on the best score
    answer =  qa.get(matched_query) #get answer to the matched query
    return answer if answer else "Sorry, Would you rephrase it?"

train("data.csv")
print("Welcome to Hotel La Vie En Rose! How can I help you?")
while True:
    print("----------------------------")
    usr_query = input("Type your query here: ")
    if usr_query.lower() == "exit":
        print("Thankyou for contacting us. Visit us soon.")
        qa=None
        emb=None
        break
    else:
        response = pred_answer(usr_query)
        print(response)