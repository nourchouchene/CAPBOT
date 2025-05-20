# ----------------- IMPORTS -----------------
import sqlite3
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import tempfile, os, base64, shutil
from gtts import gTTS
import whisper
import time
from datetime import datetime
import pandas as pd

from PIL import Image
import re


# ----------------- STREAMLIT CONFIGURATION -----------------
st.set_page_config(page_title="Assistant IA Capgemini", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
    body { background-color: #f0f8ff; }
    .stApp { font-family: 'Segoe UI', sans-serif; color: #003366; }
    .main-title { font-size: 40px; font-weight: bold; color: #003366; animation: fadeInDown 1s ease-in-out; }
    .section { margin-top: 30px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .blue-text { color: #0070ad; font-weight: bold; }
    .chat-bubble { background: #e6f2ff; border-left: 5px solid #0070ad; padding: 15px; border-radius: 10px; animation: fadeInUp 0.5s ease-out; }
    .button-main { background-color: #0070ad !important; color: white !important; }
    a { color: #0070ad; font-weight: bold; text-decoration: none; }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- DATABASE SETUP -----------------
DB_PATH = 'users.db'
CHATS_DIR = 'conversations'
os.makedirs(CHATS_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        username TEXT UNIQUE,
                        password TEXT
                    )''')
    conn.commit()
    return conn

def authenticate(username, password):
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def register(username, password):
    conn = init_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False
    
def embed_document(text, model_name="mistral"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    embeddings = OllamaEmbeddings(model=model_name)
    vectordb = FAISS.from_texts(chunks, embedding=embeddings)
    return vectordb

def process_document_and_query(doc_text, query):
    vectordb = embed_document(doc_text)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = Ollama(model="mistral")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa(query)



def save_conversation(username, chat_log):
    folder = "historiques"
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"{username}_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        for item in chat_log:
            if isinstance(item, tuple) and len(item) == 2:
                question, response = item
                f.write(f"Utilisateur : {question}\n")
                f.write(f"CapBot : {response}\n\n")
            else:
                # gestion alternative si le format est inattendu
                f.write(f"{str(item)}\n\n")

def get_conversations(username):
    folder = "historiques"
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.startswith(username)]
    return sorted(files, reverse=True)

# ----------------- TRANSCRIPTION VOCALE -----------------
def transcribe_audio(audio_file, language='fr'):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file.name, language=language)
    return result["text"]

# ----------------- ANALYSE DOCUMENT -----------------
def analyze_document(file):
    ext = file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+ext) as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    if ext == "pdf": loader = PyPDFLoader(tmp_path)
    elif ext in ["txt", "text"]: loader = TextLoader(tmp_path)
    elif ext == "docx": loader = Docx2txtLoader(tmp_path)
    elif ext in ["xlsx", "xls"]: loader = UnstructuredExcelLoader(tmp_path)
    else: return "Format non supporté"

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    qa_chain = load_qa_chain(Ollama(model="gemma:2b"), chain_type="stuff")
    query = "Fournis une analyse fonctionnelle de ce document."
    result = qa_chain.run(input_documents=texts, question=query)
    os.unlink(tmp_path)
    return result

# ----------------- AUTHENTIFICATION -----------------
import uuid
unique_conversation_id = str(uuid.uuid4())

def login_page():
    # ----------------- BACKGROUND IMAGE -----------------
    # Vous pouvez ajouter un fond ici si nécessaire, mais il est commenté pour l'instant

    # ----------------- LOGIN PAGE DESIGN -----------------
    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    # Animation du titre et autres améliorations
    st.markdown("""
    <style>
    /* Animation du titre */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Animation des champs de saisie */
    @keyframes inputFadeIn {
        0% { opacity: 0; transform: translateX(-20px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    /* Animation des boutons */
    @keyframes buttonFadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Animation de survol des boutons */
    .login-container button:hover {
        background-color: #87CEFA;
        transform: translateY(-5px);
        transition: all 0.3s ease;
    }

    /* Effet de focus sur les champs de texte */
    .login-container input:focus {
        border: 2px solid #87CEFA;
        box-shadow: 0 0 10px rgba(135, 206, 250, 0.5);
        transform: scale(1.05);
        transition: all 0.3s ease;
    }

    /* Effet de secousse pour l'erreur */
    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        50% { transform: translateX(10px); }
        75% { transform: translateX(-10px); }
        100% { transform: translateX(0); }
    }

    .error-message {
        animation: shake 0.5s ease-in-out;
        color: red;
        font-weight: bold;
    }

    /* Animation du fond de l'écran (changement de couleur) */
    @keyframes backgroundAnimation {
        0% { background-color: #ffffff; }
        50% { background-color: #87CEFA; }
        100% { background-color: #ffffff; }
    }

    body {
        animation: backgroundAnimation 10s infinite;
    }

    /* Animation de glissement des éléments */
    @keyframes slideIn {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }

    .login-header, .login-container input, .login-container button {
        animation: slideIn 0.5s ease-out;
    }

    /* Animation machine à écrire pour le titre */
    @keyframes typing {
        0% { width: 0; }
        100% { width: 100%; }
    }

    .login-header {
        font-size: 36px;
        font-weight: bold;
        background: linear-gradient(to right, #87CEFA, #ffffff);
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        text-align: center;
        margin-bottom: 30px;
        margin-top: 20px;
        overflow: hidden;
        white-space: nowrap;
        border-right: 3px solid #87CEFA;
        width: 0;
        animation: typing 3s steps(30) 1s forwards, fadeIn 1s ease-out;
    }

    /* Animation des boutons de connexion et d'inscription */
    .login-container button {
        animation: buttonFadeIn 1s ease-out;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .login-header {
            font-size: 28px;
        }

        .login-container input {
            width: 90%;
        }

        .login-container button {
            width: 90%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Titre
    st.markdown('<div class="login-header">Se connecter à CapBot</div>', unsafe_allow_html=True)

    # Champ de texte pour le nom d'utilisateur
    username = st.text_input("Nom d'utilisateur", placeholder="Entrez votre nom d'utilisateur", help="Votre nom d'utilisateur", max_chars=30)

    # Champ de texte pour le mot de passe
    password = st.text_input("Mot de passe", type="password", placeholder="Entrez votre mot de passe", help="Votre mot de passe", max_chars=30)

    # Bouton de connexion
    if st.button("Se connecter"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Connexion réussie!")
            st.session_state.show_login = False
            st.session_state.show_registration = False
            st.rerun()
        else:
            st.error("Identifiants invalides")

    # Bouton d'inscription
    if st.button("Pas de compte? Inscrivez-vous"):
        st.session_state.show_registration = True
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def registration_page():
    st.title("Inscription - CapBot")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    confirm_password = st.text_input("Confirmer le mot de passe", type="password")
    if st.button("S'inscrire"):
        if password != confirm_password:
            st.error("Les mots de passe ne correspondent pas.")
        elif register(username, password):
            st.success("Compte créé avec succès!")
            st.session_state.show_registration = False
            st.session_state.show_login = True
            st.rerun()
        else:
            st.error("Ce nom d'utilisateur existe déjà.")
def is_valid_chat_filename(filename):
    # Ne permet que les noms sûrs : lettres, chiffres, tirets, underscore, et .txt
    return re.fullmatch(r"[\w\-]+\.txt", filename) is not None

            

# ----------------- PAGE PRINCIPALE -----------------
def chatbot_page():
    if "language" not in st.session_state:
        st.session_state["language"] = "fr"

    st.sidebar.title("🔧 Navigation")
    section = st.sidebar.radio("Aller à :", ["Accueil", "Chat", "Historique", "Paramètres", "Analyse Document"])

    if section == "Accueil":
        image = Image.open("capgemini.png")
        st.image(image, width=250)
        
        st.markdown(
    """
     <div style='text-align: center; padding: 30px;'>
        <h1 style='color: #003366; font-size: 42px;'>Bienvenue dans <span style="color:#0070ad;">CapBot</span></h1>
        <p style='font-size: 18px; color: #333333; max-width: 700px; margin: 0 auto;'>
            <strong>CapBot</strong> est un assistant IA avancé conçu pour les collaborateurs de Capgemini. Il permet 
            d’analyser automatiquement des documents (PDF, Word, Excel), répondre à vos questions avec précision, 
            enregistrer l’historique de vos échanges, effectuer de la synthèse vocale, et même transcrire votre voix en texte. 
            Idéal pour gagner du temps, améliorer la productivité et faciliter l’accès à l’information métier.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

        return

    elif section == "Historique":
        st.subheader("🕓 Vos conversations")
        files = get_conversations(st.session_state.username)
        for file in files:           
            if is_valid_chat_filename(file) :
                file_path = os.path.join("historiques", file)
                if os.path.exists(file_path) :
                   with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                   st.markdown(f"### {file}")
                   st.text_area("Contenu", value=content, height=300, key=file)    
                else:
                    st.warning(f"⚠️ Le fichier est introuvable : {file}")
            else :
                     st.warning(f"⚠️ Nom de fichier invalide détecté : {file}")    


        return

    elif section == "Paramètres":
        st.subheader("🛠️ Paramètres de l'application")
        language = st.radio("Choisir la langue", ["Français", "English"])
        st.session_state.language = 'fr' if language == "Français" else 'en'
        st.success(f"Langue changée à {language}")
        return

    elif section == "Analyse Document":
         st.subheader("🔢 Analyse de document")
    uploaded_file = st.file_uploader("Charger un fichier texte, PDF, DOCX ou Excel")

    if uploaded_file is not None:
        with st.spinner("Analyse en cours..."):
            # Extraire le texte depuis le document
            full_text = analyze_document(uploaded_file)

        # Afficher un résumé ou aperçu du texte
        st.markdown(f"<div class='chat-bubble'>{full_text}</div>", unsafe_allow_html=True)

        user_question = st.text_input("❓ Pose ta question sur le document")

        if st.button("📥 Poser une question au document") and user_question:
            with st.spinner("Recherche de la réponse..."):
                # Passer le texte extrait et la question à la fonction
                result = process_document_and_query(full_text, user_question)
            st.success(result['result'])


    # Bonus : afficher les sources
    st.markdown("### 📚 Sources utilisées :")
    for doc in result["source_documents"]:
        st.write(doc.metadata)
        st.write(doc.page_content[:300])
        return

    st.markdown('<div class="main-title">🤖 Assistant IA - Capgemini</div>', unsafe_allow_html=True)
    
    if st.button("➕ Nouveau Chat"):
        st.session_state.history_log = []

    if "history_log" not in st.session_state:
        st.session_state.history_log = []


    # Afficher la conversation
    for msg in st.session_state.history_log:
        st.write(msg)

    llm = Ollama(model="gemma:2b")
    memory = ConversationBufferMemory(memory_key="history")
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""Tu es un assistant IA pour Capgemini.
Historique: {history}
Question: {input}
Réponds de façon claire et professionnelle."""
    )
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True, prompt=prompt)

    for message in st.session_state.history_log:
        st.markdown(f"<div class='chat-bubble'>{message}</div>", unsafe_allow_html=True)

    user_input = st.text_input("Posez votre question:", "")
    if st.button("Envoyer"):
        if user_input:
            st.session_state.history_log.append(f"👤 {user_input}")
            response = conversation.predict(input=user_input)
            st.session_state.history_log.append(f"🤖 {response}")
            st.markdown(f"<div class='chat-bubble'>{response}</div>", unsafe_allow_html=True)
            save_conversation(st.session_state.username, st.session_state.history_log)
            

# ----------------- LOGIC -----------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    if "show_registration" in st.session_state and st.session_state.show_registration:
        registration_page()
    else:
        login_page()
else:
    chatbot_page()