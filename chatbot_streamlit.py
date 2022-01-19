# TODOS: label quiz, slider correct/wrong classified, CAM, "Blöcke"-Grafik


# Imports
import streamlit as st
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import nltk
from nltk.stem.lancaster import LancasterStemmer

import SessionState

#from streamlit_chat import message

import json
from stopwords import worte
from random import choice


# Hier sollen die Studierenden rumspielen.
# Aktuell: Ein MLP mit einem Layer und ohne Aktivierungsfunktion
class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer1 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        out = self.layer1(x)
        return out


@st.cache(suppress_st_warning=True)
def download_punkt():
    nltk.download("punkt")
    # st.write("Finished downloading Punkt")


@st.cache(suppress_st_warning=True)
def load_data_from_json():
    # st.write("Loading data from json")
    with open("intents.json") as file:
        data = json.load(file)
    return data


# Herzstück für das Textverständnis
def bagofwords(STEMMER, s, words):
    # Input: Satz s (User-Input), Liste bekannter Wörter words
    # Output: Vektor mit Nullen und Einsen
    bag = [0 for _ in range((len(words)))]
    s_words = nltk.word_tokenize(
        s
    )  # Splitte Satz auf in einzelne Wörter und Satzzeichen
    s_words = [
        STEMMER.stem(word.lower()) for word in s_words
    ]  # "Kürze" Wörter gemäß Lancaster-Stemmer

    # Wenn Wort in Wortliste enthalten, setze 1, sonst 0
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return torch.tensor(bag).float()


@st.cache(suppress_st_warning=True)
def prepare_data(STEMMER, data):
    # st.write("Prepare data")
    words = []  # Wörter, die der Chatbot erkennen können soll
    labels = []  # zugehörige Labels (siehe Output unten)
    docs_x = []  # Trainingsgedöhns
    docs_y = []

    # Durchlaufe die Intents
    for intent in data["intents"]:
        # Speichere Pattern-Token (gekürzte Wörter) mit zugehörigen Labeln
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [
        w for w in words if not w in worte
    ]  # Schmeiße Stopwords raus (sowas wie "als" oder "habe"), die irrelevant für die Klassifizierung sind
    words = [STEMMER.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    return words, labels, docs_x, docs_y


@st.cache(suppress_st_warning=True)
def prepare_training(STEMMER, words, labels, docs_x, docs_y, device):
    # st.write("Prepare training")
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # Generiere training und output für Training des Chatbots
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [STEMMER.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    # training = torch.tensor(training).float().to(device)
    # output = torch.tensor(output).float().to(device)

    return torch.tensor(training).float().to(device), torch.tensor(output).float().to(
        device
    )


def train_chatbot_model(training, output):
    # Trainiere das Chatbot-Gehirn
    optimizer = torch.optim.Adam(st.session_state["model"].parameters(), lr=1e-3)
    loss_func = F.cross_entropy

    n_epochs = 5000
    lossliste = torch.zeros(n_epochs)

    # Da der Datensatz nur 525 Einträge enthält, brauchen wir keine Batches und können komplett trainieren
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = st.session_state["model"](training)
        loss = loss_func(out, output)
        loss.backward()
        optimizer.step()
        lossliste[epoch] = loss.item()
        # if epoch % int(n_epochs / 10) == 0:
        # st.write(epoch, loss.item())

    st.write("ChatBot finished training!")


# Wende Chatbot-Gehirn auf Nachricht an
def predict(STEMMER, message, model, words, labels, data, device):
    message = message.lower()
    result = F.softmax(model(bagofwords(STEMMER, message, words).to(device)), dim=0)
    result_index = torch.argmax(result)
    tag = labels[result_index]

    # Wie sicher ist sich der Chatbot? 0.9 ist schon ziemlich sicher.
    if result[result_index] > 0.9:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        response = choice(responses)
    else:
        # st.write("Chatbot ist sich etwas unsicher.", result[result_index].item())
        response = "Come again for Big Fudge?"
    return tag, response, result[result_index].item()


#############################################################
########################### Intro ###########################
#############################################################


def intro(session_state):
    st.write("""## 1. Einführung""")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""In dieser Lektion geht es darum, ... .""")
    with col2:
        st.image("./images/KICampusLogo.png")

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1.5])
    with col1:
        st.write("")
    with col2:
        st.write("")
    with col3:
        st.write(
            """
            ## Unterkapitel
            Auswahl über Seitenleiste

            1. Einführung
            2. Chatbot
            2. Quellen
            """
        )

    st.markdown("---")

    link = "[Notebook](https://drive.google.com/file/d/1dOWX1qQCEzhAFOPQqH7P-V_aVVmC-0e-/view?usp=sharing)"
    st.write(
        """Programmier-interessierte Lernende haben die Möglichkeit, sich hier tiefergehend mit dem dieser Lektion zugrunde liegenden Code zu befassen:"""
    )
    st.markdown(link, unsafe_allow_html=True)

    st.markdown("---")


###############################################################
########################### Testing ###########################
###############################################################


def chatbot(session_state):
    st.write("""## 2. Chatbot""")

    # download content that is needed
    download_punkt()
    # Initialisiere Stemmer
    STEMMER = LancasterStemmer()
    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data, prepare data and prepare training
    data = load_data_from_json()
    words, labels, docs_x, docs_y = prepare_data(STEMMER, data)
    training, output = prepare_training(STEMMER, words, labels, docs_x, docs_y, device)

    # if key 'model' is not in session state, initialize a classifier instance and train the chatbot model once
    if "model" not in st.session_state:
        dim_in = len(training[0])
        dim_out = len(output[0])
        st.session_state["model"] = Classifier(dim_in, dim_out).to(device)

        train_chatbot_model(training, output)

    ##########################

    # set model to eval
    st.session_state["model"].eval()

    # if "chatbot_chat" not in st.session_state:
    #    st.session_state["chatbot_chat"] = []
    # if "user_chat" not in st.session_state:
    #    st.session_state["user_chat"] = []

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []
        st.session_state["conversation"].append("Chatbot: Hi!")

    col1, col2 = st.columns(2)
    with col2:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("User:", key="user_input")
            submit = st.form_submit_button(label="Senden")

    if submit:
        string = "User: " + user_input
        st.session_state["conversation"].append(string)
        tag, response, sicher = predict(
            STEMMER, user_input, st.session_state["model"], words, labels, data, device
        )
        string = "Chatbot: " + response
        st.session_state["conversation"].append(string)
        st.session_state["conversation"].append(sicher)

        # TODO check if response auf wiedersehen -> reset conversation

    with col1:
        for entry in st.session_state["conversation"]:
            st.text(entry)


###############################################################
########################### Sources ###########################
###############################################################


def sources(session_state):
    st.write("""## Quellen""")
    st.write("""### Bilddatensätze:""")
    st.write("""TODO: Kaggle""")


#################################################################
########################### Main Page ###########################
#################################################################


session_state = SessionState.get(button_id="", slider_value=0)
st.set_page_config(page_title="KI Campus: ChaBoDocs Demo", page_icon=":pencil2:")
st.title("KI Campus: ChaBoDocs ")
st.write("## **Lektion 1: ChaBoDocs**")

st.sidebar.subheader("Unterkapitel")
PAGES = {
    "1. Einführung": intro,
    "2. Chatbot": chatbot,
    "Quellen": sources,
}
page = st.sidebar.selectbox("Auswahl:", options=list(PAGES.keys()))

st.sidebar.markdown("""---""")

st.sidebar.write("**Projektzeitraum:**")
st.sidebar.write("Juli 2021 - Februar 2022")

st.sidebar.markdown("""---""")

link = "[Notebook](https://drive.google.com/file/d/1dOWX1qQCEzhAFOPQqH7P-V_aVVmC-0e-/view?usp=sharing)"
st.sidebar.write("Python-Code zu dieser Lektion siehe")
st.sidebar.markdown(link, unsafe_allow_html=True)

st.sidebar.markdown("""---""")

link = "[KI-Campus Website](https://ki-campus.org/)"
st.sidebar.markdown(link, unsafe_allow_html=True)
st.sidebar.image("./images/KICampusLogo.png", use_column_width=True)

PAGES[page](session_state)
