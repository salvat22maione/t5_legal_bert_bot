# Usa l'ultima versione di Python 3.13
FROM python:3.11

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia tutti i file del chatbot dentro il container
COPY . /app

# Installa le dipendenze (assicurati di avere un requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Esponi la porta 5000
EXPOSE 5000

# Comando per avviare il chatbot 
CMD ["python", "app.py"]
