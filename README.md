# Legal Chatbot - European Law Q&A

## Descrizione
Questo progetto consiste in un chatbot specializzato nel diritto europeo, basato sulla logica Retrieval-Augmented Generation (RAG). Il chatbot permette agli utenti di ottenere risposte affidabili e precise riguardo a normative europee specifiche (MiCAR, CSDR e DORA).

L'applicazione è stata sviluppata utilizzando Flask per il backend, MongoDB Atlas per l'archiviazione dei dati e un modello di generazione del testo (T5-base) per la formulazione delle risposte. È distribuita tramite Azure Containers App ed è accessibile al seguente link:

🔗 [Legal Chatbot](https://legalchatbotapp.wittyfield-1bef95db.italynorth.azurecontainerapps.io/)

---

## **1. Requisiti**

Per eseguire l'applicazione in locale, è necessario avere:
- Python 3.11+
- pip
- MongoDB Atlas (database NoSQL)
- Il file `requirements.txt` fornito nel repository

---

## **2. Installazione e Setup**

### **Clonare il repository**
```sh
 git clone https://github.com/salvat22maione/t5_legal_bert_chatbot.git
 cd t5_legal_bert_chatbot.py
```

### **Creare un ambiente virtuale e installare le dipendenze**
```sh
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **Configurare il database MongoDB Atlas**
1. Crea un account su [MongoDB Atlas](https://www.mongodb.com/atlas/database).
2. Ottieni la stringa di connessione.
3. Nel file "convert_db.py" e in "app.py" inserisci la tua stringa di connessione in MONGO_URI ed esegui il codice con il comando:

```sh
python convert_db.py
```
In modo da caricare il tuo database con i dati utili al funzionamento del chatbot. Ricordati di fare in modo che il tuo database accetti richieste esterne da parte del tuo indirizzo IP
---

## **3. Avvio dell'Applicazione in Locale**

### **Avviare il server Flask**
```sh
python app.py
```
L'applicazione sarà disponibile su `http://127.0.0.1:5000/`

---
