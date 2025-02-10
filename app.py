from flask import Flask, request, render_template, Response
from transformers import AutoTokenizer, AutoModel,T5ForConditionalGeneration
from sentence_transformers import util
import time, torch, logging
from fuzzywuzzy import fuzz
from pymongo import MongoClient


# Carica il modello direttamente da Hugging Face
model_name = "tatore22/legal_bert_chatbot/qa_model"
qa_pipeline = T5ForConditionalGeneration.from_pretrained(model_name)

# Carica il modello Legal-BERT
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
# Domande generiche comuni
generic_questions = ["cos'è","cosa fa", "a cosa serve", "come funziona", "che cos'è", "puoi spiegare"]

# Configura il logging per debug
logging.basicConfig(level=logging.INFO)


# Memorizza lo storico delle conversazioni e il contesto attivo
conversation_history = []
active_context = None  # Variabile per memorizzare il contesto attuale

# Connessione a MongoDB Atlas (sostituisci con le tue credenziali)
MONGO_URI="-"
client = MongoClient(MONGO_URI)

# Seleziona il database e la collection
db = client["chatbotDB"]  # Sostituisci con il tuo database
collection = db["qa_collection"]



def is_relevant_fuzzy(query, candidate_question, threshold=75):
    """Verifica se la domanda candidata è sufficientemente simile alla query originale."""
    similarity_score = fuzz.partial_ratio(query.lower(), candidate_question.lower())
    return similarity_score >= threshold  # Restituisce True se supera la soglia

# Funzione per calcolare gli embedding con Legal-BERT
def compute_legal_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = legal_bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Media sull'ultima dimensione per l'embedding

def get_topic_from_question(query_text):
    """Identifica il topic più vicino alla domanda dell'utente"""
    topics = collection.distinct("argomento")  # Estrae tutti gli argomenti nel database
    best_match = None
    highest_similarity = 0

    for topic in topics:
        # Confronto fuzzy tra la domanda e l'argomento
        similarity = fuzz.partial_ratio(query_text.lower(), topic.lower())

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = topic

    return best_match if highest_similarity > 60 else None  # Usa una soglia di 60

def search_context_with_embeddings(query_text):
    """Cerca il contesto migliore combinando topic e embeddings"""
    # 1. Trova il topic della domanda
    topic = get_topic_from_question(query_text)

    # 2. Calcola l'embedding della query
    query_embedding = compute_legal_bert_embedding(query_text)
    relevant_contexts = []

    # 3. Filtra per argomento (se esiste un topic)
    query_filter = {"argomento": topic} if topic else {}

    for doc in collection.find(query_filter):
        # Calcola gli embedding per la domanda e il contesto
        question_embedding = compute_legal_bert_embedding(doc["domanda"])
        context_embedding = compute_legal_bert_embedding(doc["contesto"])

        # Calcola la similarità tra la query e la domanda
        question_similarity = util.pytorch_cos_sim(query_embedding, question_embedding).item()
        # Calcola la similarità tra la query e il contesto
        context_similarity = util.pytorch_cos_sim(query_embedding, context_embedding).item()

        # Similarità combinata tra domanda e contesto
        combined_similarity = 0.7 * question_similarity + 0.3 * context_similarity

        # Solo se la similarità combinata è sufficiente e la domanda è rilevante
        if combined_similarity > 0.5 and is_relevant_fuzzy(query_text, doc["domanda"]):
            relevant_contexts.append((combined_similarity, f"Domanda: {doc['domanda']} Contesto: {doc['contesto']}"))

    # Ordina i contesti rilevanti per similarità decrescente
    relevant_contexts.sort(key=lambda x: x[0], reverse=True)

    # Mostra i contesti rilevanti e le loro similarità
    print("Contesti rilevanti e le loro similarità:")
    for similarity, context in relevant_contexts:
        print(f"Similarità: {similarity:.4f} - Contesto: {context[:150]}...")

    # Restituisce i migliori 3 contesti se ci sono più di 1 contesto rilevante
    if len(relevant_contexts) == 1:
        print(f"simil: {relevant_contexts[0][0]:.4f} cont: {relevant_contexts[0][1]}")
        return [relevant_contexts[0][1]]
        

    return [context for _, context in relevant_contexts[:3]] if relevant_contexts else None



def generate_response_stream(context):
    """Genera la risposta parola per parola per lo streaming.""" 
    response = ""
    try:
        if len(context) > 1:
            # Processo ogni contesto separatamente
            for relevant_context in context:
                logging.info("contesti in for: "+ relevant_context)
                result = qa_pipeline(relevant_context, 
                                    max_length=512, 
                                    num_return_sequences=1, 
                                    num_beams=5,
                                    early_stopping=True)
                
                if not result or 'generated_text' not in result[0]:
                    logging.error("Errore nella generazione della risposta")
                    yield f"data: ⚠️ Errore nella generazione della risposta.\n\n"
                    return
                
                response += " " + result[0]['generated_text']
                logging.info(f"Aggiunta risposta: {response}")

            # Streaming della risposta parola per parola
            for word in response.split():
                yield f"data: {word}\n\n"
                time.sleep(0.3)

            yield "data: [FINE]\n\n"  # Segnale di fine risposta
        else: 
            result = qa_pipeline(context, 
                                max_length=512, 
                                num_return_sequences=1, 
                                num_beams=5,
                                early_stopping=True)
            
            if not result or 'generated_text' not in result[0]:
                logging.error("Errore nella generazione della risposta")
                yield f"data: ⚠️ Errore nella generazione della risposta.\n\n"
                return
            
            response= result[0]['generated_text']
            logging.info(f"Aggiunta risposta: {response}")

            # Streaming della risposta parola per parola
            for word in response.split():
                yield f"data: {word}\n\n"
                time.sleep(0.3)

            yield "data: [FINE]\n\n"  # Segnale di fine risposta

    except Exception as e:
        logging.error(f"Errore: {e}")
        yield f"data: ⚠️ Errore nel server. Riprova.\n\n"


# Inizializza l'app Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    global conversation_history
    # La route renderizzerà la pagina HTML di base senza form
    return render_template('index.html', conversation=conversation_history)


@app.route("/stream", methods=["GET"])
def stream():
    question = request.args.get("question", "")
    logging.info(f"Nuova richiesta di streaming con domanda: {question}")

    # Cerca il contesto migliore per la domanda
    new_context = search_context_with_embeddings(question)
    
            
    return Response(generate_response_stream(new_context), content_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True, threaded=True, use_reloader=False)

