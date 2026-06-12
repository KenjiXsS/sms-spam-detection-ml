FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora so the notebook runs offline
RUN python -c "\
import nltk; \
nltk.download('stopwords'); \
nltk.download('punkt'); \
nltk.download('punkt_tab')"

COPY . .

# Copy the versioned dataset to the canonical data/raw/ location
RUN cp notebooks/data/raw/SMSSpamCollection data/raw/SMSSpamCollection

EXPOSE 8888

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/app", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
