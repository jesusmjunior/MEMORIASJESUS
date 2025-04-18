FROM python:3.9-slim

# Configurar diretório de trabalho
WORKDIR /app

# Instalar dependências
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements para o container
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Criar estrutura de diretórios
RUN mkdir -p db memorias templates data/static

# Copiar código fonte
COPY memory_processor.py .
COPY serve.py .

# Expor porta para servidor web
EXPOSE 8080

# Comando para inicializar o aplicativo
CMD ["python", "serve.py"]
