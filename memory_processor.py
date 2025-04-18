import os
import json
import uuid
import hashlib
import datetime
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import numpy as np

# Configurações
DB_PATH = "db/jesus_chat_memorias.sqlite"
MEMORIES_DIR = "memorias"
TEMPLATES_DIR = "templates"
HTML_TEMPLATE_PATH = "app/templates/memory_template.html"

class MemoriaChatProcessor:
    """
    Processador para memórias de chat com IAs.
    Processa templates JSON e cria representações em HTML e banco de dados.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """Inicializa o processador."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(MEMORIES_DIR, exist_ok=True)
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        
        self.db_path = db_path
        self.conn = self._init_database()
    
    def _init_database(self) -> sqlite3.Connection:
        """Inicializa o banco de dados SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Criar tabela memorias
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memorias (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            title TEXT,
            model TEXT,
            language TEXT,
            summary_brief TEXT,
            problem_resolution_score REAL,
            response_completeness REAL,
            technical_accuracy REAL,
            chat_efficiency REAL,
            tags TEXT,
            filepath_csv TEXT,
            filepath_html TEXT,
            created_at TEXT
        )
        """)
        
        # Criar tabela clusters
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            name TEXT,
            keywords TEXT,
            importance REAL,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela entities
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            mentions INTEGER,
            related_clusters TEXT,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela knowledge_graph (nós)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            type TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela edges (arestas)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY,
            source TEXT,
            target TEXT,
            relationship TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (source) REFERENCES graph_nodes (id),
            FOREIGN KEY (target) REFERENCES graph_nodes (id),
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela messages
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            tokens INTEGER,
            clusters TEXT,
            sentiment TEXT,
            intent TEXT,
            key_points TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id ON clusters (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_entities ON entities (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_messages ON messages (memoria_id)")
        
        conn.commit()
        return conn
    
    def process_template(self, template_path: str) -> str:
        """
        Processa um template JSON de chat e armazena no banco de dados.
        
        Args:
            template_path: Caminho para o arquivo JSON do template
            
        Returns:
            ID da memória processada
        """
        # Carregar template
        with open(template_path, 'r', encoding='utf-8') as f:
            try:
                template = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Erro ao decodificar o template JSON: {e}")
        
        # Validar template
        self._validate_template(template)
        
        # Obter ID da memória
        memoria_id = template["metadata"]["id"]
        if not memoria_id:
            memoria_id = f"chat_{uuid.uuid4()}"
            template["metadata"]["id"] = memoria_id
        
        # Processar e salvar no banco de dados
        self._save_to_database(template)
        
        # Gerar arquivos
        csv_path = self._generate_csv_export(template)
        html_path = self._generate_html_view(template)
        
        # Atualizar caminhos de arquivo
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memorias SET filepath_csv = ?, filepath_html = ? WHERE id = ?",
            (csv_path, html_path, memoria_id)
        )
        self.conn.commit()
        
        print(f"✅ Memória processada e salva com sucesso: {memoria_id}")
        return memoria_id
    
    def _validate_template(self, template: Dict) -> bool:
        """Valida a estrutura do template."""
        required_sections = ["metadata", "semantic_structure", "conversation", "summary"]
        for section in required_sections:
            if section not in template:
                raise ValueError(f"Seção obrigatória ausente no template: {section}")
        
        # Validar metadata
        required_metadata = ["id", "title", "timestamp"]
        for field in required_metadata:
            if field not in template["metadata"]:
                template["metadata"][field] = "" if field != "timestamp" else datetime.datetime.now().isoformat()
        
        return True
    
    def _save_to_database(self, template: Dict) -> None:
        """Salva o template processado no banco de dados."""
        cursor = self.conn.cursor()
        memoria_id = template["metadata"]["id"]
        
        # Salvar memória principal
        tags = ",".join(template["metadata"].get("tags", []))
        metrics = template.get("metrics", {})
        
        cursor.execute("""
        INSERT OR REPLACE INTO memorias (
            id, timestamp, title, model, language, summary_brief, 
            problem_resolution_score, response_completeness, technical_accuracy, 
            chat_efficiency, tags, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memoria_id,
            template["metadata"].get("timestamp", ""),
            template["metadata"].get("title", ""),
            template["metadata"].get("model", ""),
            template["metadata"].get("language", ""),
            template["summary"].get("brief", ""),
            metrics.get("problem_resolution_score", 0),
            metrics.get("response_completeness", 0),
            metrics.get("technical_accuracy", 0),
            metrics.get("chat_efficiency", 0),
            tags,
            datetime.datetime.now().isoformat()
        ))
        
        # Salvar clusters
        for cluster in template["semantic_structure"].get("topic_clusters", []):
            cluster_id = cluster.get("id", f"cluster_{uuid.uuid4()}")
            keywords = ",".join(cluster.get("keywords", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO clusters (id, memoria_id, name, keywords, importance)
            VALUES (?, ?, ?, ?, ?)
            """, (
                cluster_id,
                memoria_id,
                cluster.get("name", ""),
                keywords,
                cluster.get("importance", 0)
            ))
        
        # Salvar entidades
        for entity in template["semantic_structure"].get("entities", []):
            entity_id = f"entity_{hashlib.md5(entity.get('name', '').encode()).hexdigest()[:8]}"
            related_clusters = ",".join(entity.get("related_clusters", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO entities (id, name, type, mentions, related_clusters, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entity_id,
                entity.get("name", ""),
                entity.get("type", ""),
                entity.get("mentions", 0),
                related_clusters,
                memoria_id
            ))
        
        # Salvar nós do grafo
        for node in template["semantic_structure"].get("knowledge_graph", {}).get("nodes", []):
            cursor.execute("""
            INSERT OR REPLACE INTO graph_nodes (id, label, type, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?)
            """, (
                node.get("id", ""),
                node.get("label", ""),
                node.get("type", ""),
                node.get("weight", 0),
                memoria_id
            ))
        
        # Salvar arestas do grafo
        for idx, edge in enumerate(template["semantic_structure"].get("knowledge_graph", {}).get("edges", [])):
            edge_id = f"edge_{idx}_{memoria_id}"
            cursor.execute("""
            INSERT OR REPLACE INTO graph_edges (id, source, target, relationship, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                edge.get("source", ""),
                edge.get("target", ""),
                edge.get("relationship", ""),
                edge.get("weight", 0),
                memoria_id
            ))
        
        # Salvar mensagens
        for message in template["conversation"].get("messages", []):
            message_id = message.get("id", f"msg_{uuid.uuid4()}")
            clusters = ",".join(message.get("clusters", []))
            key_points = ",".join(message.get("key_points", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO messages (
                id, memoria_id, role, content, timestamp, tokens, 
                clusters, sentiment, intent, key_points
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                memoria_id,
                message.get("role", ""),
                message.get("content", ""),
                message.get("timestamp", ""),
                message.get("tokens", 0),
                clusters,
                message.get("sentiment", ""),
                message.get("intent", ""),
                key_points
            ))
        
        self.conn.commit()
    
    def _generate_csv_export(self, template: Dict) -> str:
        """Gera um arquivo CSV da memória."""
        memoria_id = template["metadata"]["id"]
        csv_filename = f"{memoria_id}.csv"
        csv_path = os.path.join(MEMORIES_DIR, csv_filename)
        
        # Criar DataFrame para mensagens
        messages = []
        for msg in template["conversation"].get("messages", []):
            messages.append({
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
                "tokens": msg.get("tokens", 0),
                "sentiment": msg.get("sentiment", ""),
                "intent": msg.get("intent", ""),
                "key_points": ", ".join(msg.get("key_points", []))
            })
        
        df = pd.DataFrame(messages)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return csv_path
    
    def _generate_html_view(self, template: Dict) -> str:
        """Gera uma visualização HTML da memória."""
        memoria_id = template["metadata"]["id"]
        title = template["metadata"].get
      """Gera uma visualização HTML da memória."""
        memoria_id = template["metadata"]["id"]
        title = template["metadata"].get("title", "Memória de Chat")
        html_filename = f"{memoria_id}.html"
        html_path = os.path.join(MEMORIES_DIR, html_filename)
        
        # Obter dados para o template HTML
        metadata = template["metadata"]
        summary = template["summary"]
        metrics = template.get("metrics", {})
        messages = template["conversation"].get("messages", [])
        entities = template["semantic_structure"].get("entities", [])
        clusters = template["semantic_structure"].get("topic_clusters", [])
        
        # Criar HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="{metadata.get('language', 'pt-br')}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>JESUS CHAT MEMÓRIAS - {title}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet">
            <style>
                :root {
                    --primary: #0047AB;          /* Azul cobalto */
                    --primary-dark: #00008B;     /* Azul escuro */
                    --secondary: #1E90FF;        /* Azul Dodger */
                    --accent: #87CEEB;           /* Azul céu */
                    --light-blue: #E6F2FF;       /* Azul claro */
                    --dark: #000000;             /* Preto */
                    --dark-gray: #222222;        /* Cinza escuro */
                    --light: #FFFFFF;            /* Branco */
                    --gray: #F0F0F0;             /* Cinza claro */
                }
                
                body {{
                    font-family: 'Montserrat', sans-serif;
                    background-color: var(--gray);
                    color: var(--dark);
                }}
                
                .navbar {{
                    background-color: var(--dark);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }}
                
                .navbar-brand {{
                    font-weight: 800;
                    font-size: 1.4rem;
                    color: var(--light) !important;
                }}
                
                .navbar-brand .highlight {{
                    color: var(--secondary);
                }}
                
                .memory-header {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                    color: var(--light);
                    padding: 3rem 0;
                    position: relative;
                }}
                
                .memory-metadata {{
                    background-color: var(--light);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-top: -2rem;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    margin-bottom: 2rem;
                }}
                
                .metadata-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 0.5rem;
                }}
                
                .metadata-item i {{
                    color: var(--primary);
                    margin-right: 0.5rem;
                    width: 20px;
                    text-align: center;
                }}
                
                .memory-content {{
                    background-color: var(--light);
                    border-radius: 10px;
                    padding: 2rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                    margin-bottom: 2rem;
                }}
                
                .memory-title {{
                    font-weight: 800;
                    color: var(--light);
                    margin-bottom: 1rem;
                }}
                
                .model-badge {{
                    background-color: rgba(255, 255, 255, 0.2);
                    color: var(--light);
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    display: inline-block;
                    margin-bottom: 1rem;
                }}
                
                .section-title {{
                    font-weight: 700;
                    margin-bottom: 1.5rem;
                    color: var(--primary-dark);
                    border-bottom: 2px solid var(--secondary);
                    padding-bottom: 0.5rem;
                }}
                
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 1rem;
                    margin-bottom: 2rem;
                }}
                
                .metric-card {{
                    flex: 1;
                    min-width: 150px;
                    background-color: var(--light-blue);
                    border-radius: 10px;
                    padding: 1rem;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .metric-value {{
                    font-size: 2rem;
                    font-weight: 800;
                    color: var(--primary);
                }}
                
                .metric-label {{
                    font-size: 0.9rem;
                    color: var(--dark-gray);
                    font-weight: 600;
                }}
                
                .tag-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    margin-bottom: 2rem;
                }}
                
                .tag {{
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}
                
                .message {{
                    margin-bottom: 1.5rem;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                
                .message-human {{
                    background-color: var(--light-blue);
                    border-left: 5px solid var(--primary);
                }}
                
                .message-assistant {{
                    background-color: white;
                    border-left: 5px solid var(--secondary);
                }}
                
                .message-header {{
                    padding: 0.8rem 1.5rem;
                    display: flex;
                    justify-content: space-between;
                    background-color: rgba(0, 0, 0, 0.05);
                }}
                
                .message-role {{
                    font-weight: 700;
                }}
                
                .message-time {{
                    font-size: 0.8rem;
                    color: var(--dark-gray);
                }}
                
                .message-content {{
                    padding: 1.5rem;
                    white-space: pre-wrap;
                }}
                
                .message-content pre {{
                    background-color: #f8f9fa;
                    padding: 1rem;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                
                .cluster-card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .cluster-name {{
                    font-weight: 700;
                    margin-bottom: 1rem;
                    color: var(--primary-dark);
                }}
                
                .keyword {{
                    display: inline-block;
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.2rem 0.6rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    margin-right: 0.5rem;
                    margin-bottom: 0.5rem;
                }}
                
                .entity-card {{
                    display: flex;
                    align-items: center;
                    background-color: white;
                    border-radius: 10px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .entity-icon {{
                    width: 40px;
                    height: 40px;
                    background-color: var(--primary);
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 1rem;
                }}
                
                .entity-details {{
                    flex: 1;
                }}
                
                .entity-name {{
                    font-weight: 700;
                    margin-bottom: 0.3rem;
                }}
                
                .entity-type {{
                    font-size: 0.8rem;
                    color: var(--dark-gray);
                }}
                
                .entity-mentions {{
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.2rem 0.6rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}
                
                .summary-box {{
                    background-color: var(--light-blue);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 2rem;
                    border-left: 5px solid var(--primary);
                }}
                
                .insight-item {{
                    margin-bottom: 1rem;
                    padding-left: 1.5rem;
                    position: relative;
                }}
                
                .insight-item::before {{
                    content: "•";
                    color: var(--primary);
                    font-weight: bold;
                    position: absolute;
                    left: 0;
                }}
                
                .action-item {{
                    margin-bottom: 1rem;
                    padding-left: 1.5rem;
                    position: relative;
                }}
                
                .action-item::before {{
                    content: "✓";
                    color: var(--primary);
                    font-weight: bold;
                    position: absolute;
                    left: 0;
                }}
                
                .cross {{
                    width: 40px;
                    height: 40px;
                    margin-bottom: 1rem;
                    position: relative;
                }}
                
                .cross span {{
                    position: absolute;
                    background-color: var(--secondary);
                    border-radius: 2px;
                }}
                
                .cross span:nth-child(1) {{
                    width: 4px;
                    height: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                }}
                
                .cross span:nth-child(2) {{
                    width: 100%;
                    height: 4px;
                    top: 50%;
                    transform: translateY(-50%);
                }}
                
                .footer {{
                    background-color: var(--dark);
                    color: var(--light);
                    padding: 2rem 0;
                    text-align: center;
                }}
                
                .code {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 1rem;
                    overflow-x: auto;
                    font-family: monospace;
                    margin-bottom: 1rem;
                }}
            </style>
        </head>
        <body>
            <!-- Navbar -->
            <nav class="navbar navbar-dark">
                <div class="container">
                    <a class="navbar-brand" href="../index.html">
                        <span class="highlight">JESUS</span> CHAT MEMÓRIAS
                    </a>
                    <div>
                        <a href="../index.html" class="btn btn-outline-light btn-sm">
                            <i class="fas fa-home me-1"></i> Home
                        </a>
                    </div>
                </div>
            </nav>
            
            <!-- Memory Header -->
            <header class="memory-header">
                <div class="container">
                    <div class="model-badge">
                        <i class="fas fa-robot me-1"></i> {metadata.get('model', 'IA')}
                    </div>
                    <h1 class="memory-title">{title}</h1>
                    <p class="lead text-white opacity-75">
                        <i class="far fa-calendar-alt me-1"></i> {self._format_date(metadata.get('timestamp', ''))}
                    </p>
                </div>
            </header>
            
            <!-- Main Content -->
            <div class="container py-4">
                <!-- Metadata -->
                <div class="memory-metadata">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-tag"></i>
                                <span>ID: {memoria_id}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-language"></i>
                                <span>Idioma: {metadata.get('language', 'Não especificado')}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-comment-dots"></i>
                                <span>Mensagens: {len(messages)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-clock"></i>
                                <span>Duração: {metadata.get('duration_seconds', 0)} segundos</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-calculator"></i>
                                <span>Tokens: {metadata.get('total_tokens', 0)}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-server"></i>
                                <span>Plataforma: {metadata.get('platform', 'Não especificada')}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Summary -->
                <div class="memory-content">
                    <h2 class="section-title">Resumo</h2>
                    <div class="summary-box">
                        <p class="mb-0">{summary.get('brief', 'Não há resumo disponível.')}</p>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h3 class="section-title h5">Insights Principais</h3>
                            <div class="insights-container">
                                {self._generate_insights_html(summary.get('key_insights', []))}
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h3 class="section-title h5">Ações Recomendadas</h3>
                            <div class="actions-container">
                                {self._generate_actions_html(summary.get('action_items', []))}
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Metrics -->
                <div class="memory-content">
                    <h2 class="section-title">Métricas</h2>
                    <div class="metrics-container">
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('problem_resolution_score', 0))}</div>
                            <div class="metric-label">Resolução</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('technical_accuracy', 0))}</div>
                            <div class="metric-label">Precisão</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('response_completeness', 0))}</div>
                            <div class="metric-label">Completude</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('chat_efficiency', 0))}</div>
                            <div class="metric-label">Eficiência</div>
                        </div>
                    </div>
                </div>
                
                <!-- Tags -->
                <div class="memory-content">
                    <h2 class="section-title">Tags</h2>
                    <div class="tag-container">
                        {self._generate_tags_html(metadata.get('tags', []))}
                    </div>
                </div>
                
                <!-- Clusters -->
                <div class="memory-content">
                    <h2 class="section-title">Clusters Temáticos</h2>
                    <div class="clusters-container">
                        {self._generate_clusters_html(clusters)}
                    </div>
                </div>
                
                <!-- Entities -->
                <div class="memory-content">
                    <h2 class="section-title">Entidades</h2>
                    <div class="entities-container">
                        {self._generate_entities_html(entities)}
                    </div>
                </div>
                
                <!-- Chat -->
                <div class="memory-content">
                    <h2 class="section-title">Conversa</h2>
                    <div class="conversation-container">
                        {self._generate_messages_html(messages)}
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <footer class="footer">
                <div class="container">
                    <div class="cross">
                        <span></span>
                        <span></span>
                    </div>
                    <p class="mb-0">JESUS CHAT MEMÓRIAS &copy; {datetime.datetime.now().year}</p>
                    <small>Memória gerada em {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}</small>
                </div>
            </footer>
        </body>
        </html>
        """
        
        # Salvar arquivo HTML
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _format_date(self, timestamp: str) -> str:
        """Formata timestamp para exibição."""
        if not timestamp:
            return "Data não especificada"
        
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%d/%m/%Y %H:%M')
        except (ValueError, TypeError):
            return timestamp
    
    def _format_percentage(self, value: float) -> str:
        """Formata valor de métrica como porcentagem."""
        if not value:
            return "0%"
        
        return f"{int(value * 100)}%"
    
    def _generate_insights_html(self, insights: List[str]) -> str:
        """Gera HTML para insights."""
        if not insights:
            return "<p>Nenhum insight disponível.</p>"
        
        html = ""
        for insight in insights:
            html += f'<div class="insight-item">{insight}</div>'
        
        return html
    
    def _generate_actions_html(self, actions: List[str]) -> str:
        """Gera HTML para ações recomendadas."""
        if not actions:
            return "<p>Nenhuma ação recomendada.</p>"
        
        html = ""
        for action in actions:
            html += f'<div class="action-item">{action}</div>'
        
        return html
    
    def _generate_tags_html(self, tags: List[str]) -> str:
        """Gera HTML para tags."""
        if not tags:
            return "<span class='tag'>sem-tags</span>"
        
        html = ""
        for tag in tags:
            html += f'<span class="tag">{tag}</span>'
        
        return html
    
    def _generate_clusters_html(self, clusters: List[Dict]) -> str:
        """Gera HTML para clusters temáticos."""
        if not clusters:
            return "<p>Nenhum cluster temático identificado.</p>"
        
        html = ""
        for cluster in clusters:
            keywords_html = ""
            for keyword in cluster.get("keywords", []):
                keywords_html += f'<span class="keyword">{keyword}</span>'
            
            importance = cluster.get("importance", 0)
            importance_percent = self._format_percentage(importance)
            
            html += f"""
            <div class="cluster-card">
                <div class="d-flex justify-content-between align-items-center">
                    <h3 class="cluster-name">{cluster.get("name", "Sem nome")}</h3>
                    <span class="entity-mentions">Relevância: {importance_percent}</span>
                </div>
                <div class="keywords-container">
                    {keywords_html}
                </div>
            </div>
            """
        
        return html
    
    def _generate_entities_html(self, entities: List[Dict]) -> str:
        """Gera HTML para entidades."""
        if not entities:
            return "<p>Nenhuma entidade identificada.</p>"
        
        html = ""
        for entity in entities:
            entity_type = entity.get("type", "conceito")
            icon = self._get_entity_icon(entity_type)
            
            html += f"""
            <div class="entity-card">
                <div class="entity-icon">
                    <i class="{icon}"></i>
                </div>
                <div class="entity-details">
                    <div class="entity-name">{entity.get("name", "Sem nome")}</div>
                    <div class="entity-type">Tipo: {entity_type}</div>
                </div>
                <span class="entity-mentions">{entity.get("mentions", 0)} menções</span>
            </div>
            """
        
        return html
    
    def _get_entity_icon(self, entity_type: str) -> str:
        """Retorna ícone baseado no tipo de entidade."""
        icons = {
            "technology": "fas fa-microchip",
            "language": "fas fa-code",
            "concept": "fas fa-lightbulb",
            "person": "fas fa-user",
            "organization": "fas fa-building",
            "location": "fas fa-map-marker-alt",
            "protocol": "fas fa-shield-alt",
            "application": "fas fa-cube",
            "framework": "fas fa-layer-group",
            "architecture": "fas fa-sitemap"
        }
        
        return icons.get(entity_type.lower(), "fas fa-tag")
    
    def _generate_messages_html(self, messages: List[Dict]) -> str:
        """Gera HTML para mensagens."""
        if not messages:
            return "<p>Nenhuma mensagem disponível.</p>"
        
        html = ""
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = self._format_date(message.get("timestamp", ""))
            
            # Formatação de código em blocos de código
            content = self._format_code_blocks(content)
            
            message_class = "message-human" if role == "human" else "message-assistant"
            role_display = "Usuário" if role == "human" else "Assistente"
            
            html += f"""
            <div class="message {message_class}">
                <div class="message-header">
                    <div class="message-role">{role_display}</div>
                    <div class="message-time">{timestamp}</div>
                </div>
                <div class="message-content">
                    {content}
                </div>
            </div>
            """
        
        return html
    
    def _format_code_blocks(self, content: str) -> str:
        """Formata blocos de código no conteúdo."""
        # Padrão para corresponder a blocos de código markdown
        pattern = r'```(.*?)\n(.*?)```'
        
        def replace_code_block(match):
            language = match.group(1).strip()
            code = match.group(2)
            return f'<pre><code class="language-{language}">{code}</code></pre>'
        
        # Substituir blocos de código
        formatted_content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)
        
        return formatted_content
    
    def search_chats(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats com base em uma consulta textual.
        
        Args:
            query: Texto para busca
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        # Busca por título, resumo e tags
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        WHERE 
            m.title LIKE ? OR 
            m.summary_brief LIKE ? OR 
            m.tags LIKE ?
        ORDER BY m.timestamp DESC
        LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def search_by_entity(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats que mencionam uma entidade específica.
        
        Args:
            entity_name: Nome da entidade
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        JOIN entities e ON m.id = e.memoria_id
        WHERE e.name LIKE ?
        ORDER BY e.mentions DESC
        LIMIT ?
        """, (f"%{entity_name}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def search_by_cluster(self, cluster_topic: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats relacionados a um cluster/tópico específico.
        
        Args:
            cluster_topic: Tópico ou palavra-chave do cluster
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        JOIN clusters c ON m.id = c.memoria_id
        WHERE c.name LIKE ? OR c.keywords LIKE ?
        ORDER BY c.importance DESC
        LIMIT ?
        """, (f"%{cluster_topic}%", f"%{cluster_topic}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def export_to_csv(self, output_path: str = "jesus_chat_memorias_export.csv") -> str:
        """
        Exporta o banco de dados de memórias para um arquivo CSV.
        
        Args:
            output_path: Caminho para salvar o arquivo CSV
            
        Returns:
            Caminho do arquivo CSV exportado
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT 
            m.id, m.timestamp, m.title, m.model, m.language, m.import os
import json
import uuid
import hashlib
import datetime
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import numpy as np

# Configurações
DB_PATH = "db/jesus_chat_memorias.sqlite"
MEMORIES_DIR = "memorias"
TEMPLATES_DIR = "templates"
HTML_TEMPLATE_PATH = "app/templates/memory_template.html"

class MemoriaChatProcessor:
    """
    Processador para memórias de chat com IAs.
    Processa templates JSON e cria representações em HTML e banco de dados.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """Inicializa o processador."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(MEMORIES_DIR, exist_ok=True)
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        
        self.db_path = db_path
        self.conn = self._init_database()
    
    def _init_database(self) -> sqlite3.Connection:
        """Inicializa o banco de dados SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Criar tabela memorias
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memorias (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            title TEXT,
            model TEXT,
            language TEXT,
            summary_brief TEXT,
            problem_resolution_score REAL,
            response_completeness REAL,
            technical_accuracy REAL,
            chat_efficiency REAL,
            tags TEXT,
            filepath_csv TEXT,
            filepath_html TEXT,
            created_at TEXT
        )
        """)
        
        # Criar tabela clusters
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            name TEXT,
            keywords TEXT,
            importance REAL,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela entities
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            mentions INTEGER,
            related_clusters TEXT,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela knowledge_graph (nós)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            type TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela edges (arestas)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY,
            source TEXT,
            target TEXT,
            relationship TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (source) REFERENCES graph_nodes (id),
            FOREIGN KEY (target) REFERENCES graph_nodes (id),
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela messages
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            tokens INTEGER,
            clusters TEXT,
            sentiment TEXT,
            intent TEXT,
            key_points TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id ON clusters (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_entities ON entities (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_messages ON messages (memoria_id)")
        
        conn.commit()
        return conn
    
    def process_template(self, template_path: str) -> str:
        """
        Processa um template JSON de chat e armazena no banco de dados.
        
        Args:
            template_path: Caminho para o arquivo JSON do template
            
        Returns:
            ID da memória processada
        """
        # Carregar template
        with open(template_path, 'r', encoding='utf-8') as f:
            try:
                template = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Erro ao decodificar o template JSON: {e}")
        
        # Validar template
        self._validate_template(template)
        
        # Obter ID da memória
        memoria_id = template["metadata"]["id"]
        if not memoria_id:
            memoria_id = f"chat_{uuid.uuid4()}"
            template["metadata"]["id"] = memoria_id
        
        # Processar e salvar no banco de dados
        self._save_to_database(template)
        
        # Gerar arquivos
        csv_path = self._generate_csv_export(template)
        html_path = self._generate_html_view(template)
        
        # Atualizar caminhos de arquivo
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memorias SET filepath_csv = ?, filepath_html = ? WHERE id = ?",
            (csv_path, html_path, memoria_id)
        )
        self.conn.commit()
        
        print(f"✅ Memória processada e salva com sucesso: {memoria_id}")
        return memoria_id
    
    def _validate_template(self, template: Dict) -> bool:
        """Valida a estrutura do template."""
        required_sections = ["metadata", "semantic_structure", "conversation", "summary"]
        for section in required_sections:
            if section not in template:
                raise ValueError(f"Seção obrigatória ausente no template: {section}")
        
        # Validar metadata
        required_metadata = ["id", "title", "timestamp"]
        for field in required_metadata:
            if field not in template["metadata"]:
                template["metadata"][field] = "" if field != "timestamp" else datetime.datetime.now().isoformat()
        
        return True
    
    def _save_to_database(self, template: Dict) -> None:
        """Salva o template processado no banco de dados."""
        cursor = self.conn.cursor()
        memoria_id = template["metadata"]["id"]
        
        # Salvar memória principal
        tags = ",".join(template["metadata"].get("tags", []))
        metrics = template.get("metrics", {})
        
        cursor.execute("""
        INSERT OR REPLACE INTO memorias (
            id, timestamp, title, model, language, summary_brief, 
            problem_resolution_score, response_completeness, technical_accuracy, 
            chat_efficiency, tags, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memoria_id,
            template["metadata"].get("timestamp", ""),
            template["metadata"].get("title", ""),
            template["metadata"].get("model", ""),
            template["metadata"].get("language", ""),
            template["summary"].get("brief", ""),
            metrics.get("problem_resolution_score", 0),
            metrics.get("response_completeness", 0),
            metrics.get("technical_accuracy", 0),
            metrics.get("chat_efficiency", 0),
            tags,
            datetime.datetime.now().isoformat()
        ))
        
        # Salvar clusters
        for cluster in template["semantic_structure"].get("topic_clusters", []):
            cluster_id = cluster.get("id", f"cluster_{uuid.uuid4()}")
            keywords = ",".join(cluster.get("keywords", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO clusters (id, memoria_id, name, keywords, importance)
            VALUES (?, ?, ?, ?, ?)
            """, (
                cluster_id,
                memoria_id,
                cluster.get("name", ""),
                keywords,
                cluster.get("importance", 0)
            ))
        
        # Salvar entidades
        for entity in template["semantic_structure"].get("entities", []):
            entity_id = f"entity_{hashlib.md5(entity.get('name', '').encode()).hexdigest()[:8]}"
            related_clusters = ",".join(entity.get("related_clusters", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO entities (id, name, type, mentions, related_clusters, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entity_id,
                entity.get("name", ""),
                entity.get("type", ""),
                entity.get("mentions", 0),
                related_clusters,
                memoria_id
            ))
        
        # Salvar nós do grafo
        for node in template["semantic_structure"].get("knowledge_graph", {}).get("nodes", []):
            cursor.execute("""
            INSERT OR REPLACE INTO graph_nodes (id, label, type, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?)
            """, (
                node.get("id", ""),
                node.get("label", ""),
                node.get("type", ""),
                node.get("weight", 0),
                memoria_id
            ))
        
        # Salvar arestas do grafo
        for idx, edge in enumerate(template["semantic_structure"].get("knowledge_graph", {}).get("edges", [])):
            edge_id = f"edge_{idx}_{memoria_id}"
            cursor.execute("""
            INSERT OR REPLACE INTO graph_edges (id, source, target, relationship, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                edge.get("source", ""),
                edge.get("target", ""),
                edge.get("relationship", ""),
                edge.get("weight", 0),
                memoria_id
            ))
        
        # Salvar mensagens
        for message in template["conversation"].get("messages", []):
            message_id = message.get("id", f"msg_{uuid.uuid4()}")
            clusters = ",".join(message.get("clusters", []))
            key_points = ",".join(message.get("key_points", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO messages (
                id, memoria_id, role, content, timestamp, tokens, 
                clusters, sentiment, intent, key_points
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                memoria_id,
                message.get("role", ""),
                message.get("content", ""),
                message.get("timestamp", ""),
                message.get("tokens", 0),
                clusters,
                message.get("sentiment", ""),
                message.get("intent", ""),
                key_points
            ))
        
        self.conn.commit()
    
    def _generate_csv_export(self, template: Dict) -> str:
        """Gera um arquivo CSV da memória."""
        memoria_id = template["metadata"]["id"]
        csv_filename = f"{memoria_id}.csv"
        csv_path = os.path.join(MEMORIES_DIR, csv_filename)
        
        # Criar DataFrame para mensagens
        messages = []
        for msg in template["conversation"].get("messages", []):
            messages.append({
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
                "tokens": msg.get("tokens", 0),
                "sentiment": msg.get("sentiment", ""),
                "intent": msg.get("intent", ""),
                "key_points": ", ".join(msg.get("key_points", []))
            })
        
        df = pd.DataFrame(messages)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return csv_path
    
    def _generate_html_view(self, template: Dict) -> str:
        """Gera uma visualização HTML da memória."""
        memoria_id = template["metadata"]["id"]
        title = template["metadata"].get
        m.id, m.timestamp, m.title, m.model, m.language, m.summary_brief, 
            m.problem_resolution_score, m.response_completeness, m.technical_accuracy, 
            m.chat_efficiency, m.tags, m.filepath_csv, m.filepath_html, m.created_at
        FROM memorias m
        """)
        
        results = cursor.fetchall()
        
        # Criar DataFrame para exportação
        columns = [
            "id", "timestamp", "title", "model", "language", "summary_brief",
            "problem_resolution_score", "response_completeness", "technical_accuracy",
            "chat_efficiency", "tags", "filepath_csv", "filepath_html", "created_at"
        ]
        
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path
    
    def generate_graph_data(self) -> Dict:
        """
        Gera dados para visualização do grafo de conhecimento.
        
        Returns:
            Dicionário com nós e arestas para visualização
        """
        cursor = self.conn.cursor()
        
        # Buscar todos os nós
        cursor.execute("""
        SELECT id, label, type, weight, memoria_id FROM graph_nodes
        """)
        nodes_data = cursor.fetchall()
        
        # Buscar todas as arestas
        cursor.execute("""
        SELECT source, target, relationship, weight FROM graph_edges
        """)
        edges_data = cursor.fetchall()
        
        # Formatar para visualização
        nodes = []
        for node in nodes_data:
            nodes.append({
                "id": node[0],
                "label": node[1],
                "type": node[2],
                "weight": node[3],
                "memoria_id": node[4]
            })
        
        edges = []
        for edge in edges_data:
            edges.append({
                "source": edge[0],
                "target": edge[1],
                "relationship": edge[2],
                "weight": edge[3]
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def generate_word_cloud_data(self) -> Dict:
        """
        Gera dados para nuvem de palavras baseada em entidades e keywords.
        
        Returns:
            Dicionário com palavras e suas contagens
        """
        cursor = self.conn.cursor()
        
        # Buscar entidades e suas menções
        cursor.execute("""
        SELECT name, SUM(mentions) as total_mentions FROM entities
        GROUP BY name
        ORDER BY total_mentions DESC
        LIMIT 100
        """)
        entity_data = cursor.fetchall()
        
        # Buscar keywords dos clusters
        cursor.execute("""
        SELECT keywords FROM clusters
        """)
        keywords_data = cursor.fetchall()
        
        # Processar entidades
        word_counts = {}
        for entity in entity_data:
            word_counts[entity[0]] = int(entity[1])
        
        # Processar keywords
        for keywords_row in keywords_data:
            if keywords_row[0]:
                keywords = keywords_row[0].split(',')
                for keyword in keywords:
                    if keyword:
                        word_counts[keyword] = word_counts.get(keyword, 0) + 5  # Peso adicional para keywords
        
        # Formatar resultado
        word_cloud_data = [{"text": word, "value": count} for word, count in word_counts.items()]
        
        return {
            "words": word_cloud_data
        }
    
    def get_stats(self) -> Dict:
        """
        Obtém estatísticas gerais do banco de dados.
        
        Returns:
            Dicionário com estatísticas
        """
        cursor = self.conn.cursor()
        
        # Total de memórias
        cursor.execute("SELECT COUNT(*) FROM memorias")
        total_memorias = cursor.fetchone()[0]
        
        # Total de entidades
        cursor.execute("SELECT COUNT(*) FROM entities")
        total_entities = cursor.fetchone()[0]
        
        # Total de clusters
        cursor.execute("SELECT COUNT(*) FROM clusters")
        total_clusters = cursor.fetchone()[0]
        
        # Total de mensagens
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        # Modelos utilizados
        cursor.execute("""
        SELECT model, COUNT(*) as count FROM memorias
        GROUP BY model
        ORDER BY count DESC
        """)
        models_data = cursor.fetchall()
        models = {model: count for model, count in models_data}
        
        # Tags mais comuns
        cursor.execute("""
        SELECT tags FROM memorias
        """)
        tags_rows = cursor.fetchall()
        
        tag_counts = {}
        for row in tags_rows:
            if row[0]:
                tags = row[0].split(',')
                for tag in tags:
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            "total_memorias": total_memorias,
            "total_entities": total_entities,
            "total_clusters": total_clusters,
            "total_messages": total_messages,
            "models": models,
            "top_tags": top_tags,
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self.conn:
            self.conn.close()


def process_directory(processor: MemoriaChatProcessor, directory: str) -> List[str]:
    """
    Processa todos os arquivos JSON em um diretório.
    
    Args:
        processor: Instância do processador
        directory: Caminho do diretório com templates JSON
        
    Returns:
        Lista de IDs das memórias processadas
    """
    processed_ids = []
    
    # Listar arquivos JSON no diretório
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            memoria_id = processor.process_template(file_path)
            processed_ids.append(memoria_id)
            print(f"Processado: {json_file} → ID: {memoria_id}")
        except Exception as e:
            print(f"Erro ao processar {json_file}: {e}")
    
    return processed_ids


def generate_index_html(processor: MemoriaChatProcessor, output_path: str = "index.html") -> str:
    """
    Gera página HTML inicial com dashboard.
    
    Args:
        processor: Instância do processador
        output_path: Caminho para salvar o arquivo HTML
        
    Returns:
        Caminho do arquivo HTML gerado
    """
    # Obter estatísticas
    stats = processor.get_stats()
    
    # Obter dados para nuvem de palavras
    word_cloud_data = processor.generate_word_cloud_data()
    
    # Converter para JSON para uso no JavaScript
    stats_json = json.dumps(stats)
    word_cloud_json = json.dumps(word_cloud_data)
    
    # Cursor para obter memórias recentes
    cursor = processor.conn.cursor()
    cursor.execute("""
    SELECT id, title, model, timestamp, summary_brief, tags, filepath_html
    FROM memorias
    ORDER BY timestamp DESC
    LIMIT 6
    """)
    
    recent_memories = cursor.fetchall()
    
    # Criar HTML da página inicial
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JESUS CHAT MEMÓRIAS</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        <script src="https://cdn.jsdelivr.net/npm/d3-cloud@1.2.5/build/d3.layout.cloud.min.js"></script>
        <style>
            :root {
                --primary: #0047AB;          /* Azul cobalto */
                --primary-dark: #00008B;     /* Azul escuro */
                --secondary: #1E90FF;        /* Azul Dodger */
                --accent: #87CEEB;           /* Azul céu */
                --light-blue: #E6F2FF;       /* Azul claro */
                --dark: #000000;             /* Preto */
                --dark-gray: #222222;        /* Cinza escuro */
                --light: #FFFFFF;            /* Branco */
                --gray: #F0F0F0;             /* Cinza claro */
            }
            
            body {
                font-family: 'Montserrat', sans-serif;
                background-color: var(--gray);
                color: var(--dark);
            }
            
            /* Navbar */
            .navbar {
                background-color: var(--dark);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            
            .navbar-brand {
                font-weight: 800;
                font-size: 1.4rem;
                color: var(--light) !important;
            }
            
            .navbar-brand .highlight {
                color: var(--secondary);
            }
            
            .nav-link {
                color: var(--light) !important;
                font-weight: 600;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                margin: 0 0.2rem;
                border-radius: 4px;
            }
            
            .nav-link:hover, 
            .nav-link.active {
                background-color: var(--primary);
                color: var(--light) !important;
            }
            
            /* Hero Section */
            .hero-section {
                background: linear-gradient(135deg, var(--dark) 0%, var(--primary-dark) 100%);
                color: var(--light);
                padding: 5rem 0;
                position: relative;
                overflow: hidden;
            }
            
            .hero-title {
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 1.5rem;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            }
            
            .hero-subtitle {
                font-size: 1.5rem;
                margin-bottom: 2rem;
                opacity: 0.9;
            }
            
            .cross-icon {
                font-size: 5rem;
                color: var(--secondary);
                margin-bottom: 2rem;
                text-shadow: 0 0 20px rgba(30, 144, 255, 0.7);
            }
            
            /* Stats Cards */
            .stats-section {
                margin-top: -60px;
                z-index: 10;
                position: relative;
                padding-bottom: 3rem;
            }
            
            .stats-card {
                background: var(--light);
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                text-align: center;
                transition: all 0.3s ease;
                border-top: 5px solid var(--primary);
                margin-bottom: 1.5rem;
                height: 100%;
            }
            
            .stats-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
            }
            
            .stats-icon {
                height: 80px;
                width: 80px;
                background: var(--light-blue);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1.5rem;
                color: var(--primary);
                font-size: 2rem;
                box-shadow: 0 5px 15px rgba(0, 71, 171, 0.2);
            }
            
            .stats-value {
                font-size: 2.5rem;
                font-weight: 800;
                color: var(--primary);
                margin-bottom: 0.5rem;
            }
            
            .stats-label {
                font-size: 1.1rem;
                color: var(--dark-gray);
                font-weight: 600;
            }
            
            /* Content Sections */
            .content-section {
                padding: 5rem 0;
            }
            
            .content-section.bg-light {
                background-color: var(--light);
            }
            
            .section-title {
                font-size: 2rem;
                font-weight: 800;
                text-align: center;
                margin-bottom: 1rem;
                color: var(--dark);
            }
            
            .section-subtitle {
                font-size: 1.2rem;
                text-align: center;
                margin-bottom: 3rem;
                color: var(--primary);
            }
            
            /* Memory Cards */
            .memory-card {
                background: var(--light);
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
                height: 100%;
                display: flex;
                flex-direction: column;
                margin-bottom: 2rem;
            }
            
            .memory-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
            }
            
            .memory-header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: var(--light);
                padding: 1.5rem;
                border-bottom: 3px solid var(--secondary);
            }
            
            .memory-model {
                display: inline-block;
                background: rgba(255, 255, 255, 0.2);
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .memory-title {
                font-size: 1.3rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .memory-date {
                font-size: 0.9rem;
                opacity: 0.8;
            }
            
            .memory-body {
                padding: 1.5rem;
                flex-grow: 1;
            }
            
            .memory-summary {
                color: var(--dark-gray);
                margin-bottom: 1.5rem;
                font-size: 0.95rem;
                line-height: 1.6;
                display: -webkit-box;
                -webkit-line-clamp: 4;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            
            .memory-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .memory-tag {
                display: inline-block;
                padding: 0.3rem 0.8rem;
                background: var(--light-blue);
                border-radius: 20px;
                font-size: 0.8rem;
                color: var(--primary-dark);
                font-weight: 600;
            }
            
            .memory-footer {
                padding: 1.5rem;
                border-top: 1px solid var(--gray);
            }
            
            .btn-view {
                background-color: var(--primary);
                color: var(--light);
                border: none;
                border-radius: 30px;
                padding: 0.5rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .btn-view:hover {
                background-color: var(--primary-dark);
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            /* Charts */
            .chart-container {
                background: var(--light);
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
                height: 100%;
            }
            
            .word-cloud-container {
                height: 400px;
                position: relative;
            }
            
            /* Footer */
            .footer {
                background-color: var(--dark);
                color: var(--light);
                padding: 4rem 0 2rem;
            }
            
            .footer-logo {
                font-size: 1.8rem;
                font-weight: 800;
                margin-bottom: 1rem;
            }
            
            .footer-logo .highlight {
                color: var(--secondary);
            }
            
            .cross {
                position: relative;
                width: 60px;
                height: 60px;
                margin: 0 auto 2rem;
            }
            
            .cross span {
                position: absolute;
                background-color: var(--secondary);
                border-radius: 4px;
            }
            
            .cross span:nth-child(1) {
                width: 4px;
                height: 100%;
                left: 50%;
                transform: translateX(-50%);
                animation: pulse 2s infinite;
            }
            
            .cross span:nth-child(2) {
                width: 100%;
                height: 4px;
                top: 50%;
                transform: translateY(-50%);
                animation: pulse 2s infinite 0.3s;
            }
            
            @keyframes pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(30, 144, 255, 0.7);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(30, 144, 255, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(30, 144, 255, 0);
                }
            }
            
            .search-container {
                background: var(--light);
                border-radius: 30px;
                overflow: hidden;
                display: flex;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            
            .search-input {
                flex: 1;
                border: none;
                padding: 1rem 1.5rem;
                font-size: 1rem;
            }
            
            .search-input:focus {
                outline: none;
            }
            
            .search-button {
                background: var(--primary);
                color: white;
                border: none;
                padding: 0 1.5rem;
                font-size: 1.2rem;
            }
            
            .search-button:hover {
                background: var(--primary-dark);
            }
        </style>
    </head>
    <body>
        <!-- Navbar -->
        <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
            <div class="container">
                <a class="navbar-brand" href="index.html">
                    <span class="highlight">JESUS</span> CHAT MEMÓRIAS
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav ms-auto">
                        <li class="nav-item">
                            <a class="nav-link active" href="index.html">
                                <i class="fas fa-home me-1"></i> Home
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="search.html">
                                <i class="fas fa-search me-1"></i> Buscar
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="entities.html">
                                <i class="fas fa-tags me-1"></i> Entidades
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="graph.html">
                                <i class="fas fa-project-diagram me-1"></i> Grafo
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="upload.html">
                                <i class="fas fa-upload me-1"></i> Upload
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <!-- Hero Section -->
        <section class="hero-section">
            <div class="container text-center">
                <div class="cross">
                    <span></span>
                    <span></span>
                </div>
                <h1 class="hero-title">JESUS CHAT MEMÓRIAS</h1>
                <p class="hero-subtitle">Sistema de Indexação para Conversas com IAs</p>
                <div class="search-container mx-auto" style="max-width: 600px;">
                    <input type="text" class="search-input" placeholder="Buscar por assunto, entidade ou tag...">
                    <button class="search-button">
                        <i class="fas fa-search"></i>
                    </button>
                </div>
            </div>
        </section>
        
        <!-- Stats Section -->
        <section class="stats-section">
            <div class="container">
                <div class="row">
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-comments"></i>
                            </div>
                            <div class="stats-value" id="memoriasCount">{stats['total_memorias']}</div>
                            <div class="stats-label">Memórias</div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-sitemap"></i>
                            </div>
                            <div class="stats-value" id="clustersCount">{stats['total_clusters']}</div>
                            <div class="stats-label">Clusters</div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-tag"></i>
                            </div>
                            <div class="stats-value" id="entitiesCount">{stats['total_entities']}</div>
                            <div class="stats-label">Entidades</div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-comment-dots"></i>
                            </div>
                            <div class="stats-value" id="messagesCount">{stats['total_messages']}</div>
                            <div class="stats-label">Mensagens</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Charts Section -->
        <section class="content-section bg-light">
            <div class="container">
                <h2 class="section-title">Análise de Dados</h2>
                <p class="section-subtitle">Visualizações e métricas das suas conversas com IAs</p>
                
                <div class="row">
                    <div class="col-lg-6 mb-4">
                        <div class="chart-container">
                            <h3 class="h5 mb-4">Distribuição de Modelos</h3>
                            <canvas id="modelsChart"></canvas>
                        </div>
                    </div>
                    <div class="col-lg-6 mb-4">
                        <div class="chart-container">
                            <h3 class="h5 mb-4">Tags Mais Comuns</h3>
                            <canvas id="tagsChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12 mb-4">
                        <div class="chart-container">
                            <h3 class="h5 mb-4">Nuvem de Palavras</h3>
                            <div class="word-cloud-container" id="wordCloud"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Recent Memories Section -->
        <section class="content-section">
            <div class="container">
                <h2 class="section-title">Memórias Recentes</h2>
                <p class="section-subtitle">Últimas conversas processadas</p>
                
                <div class="row">
                    {self._generate_memory_cards_html(recent_memories)}
                </div>
                
                <div class="text-center mt-4">
                    <a href="search.html" class="btn btn-primary btn-lg">
                        <i class="fas fa-search me-2"></i>Ver Todas as Memórias
                    </a>
                </div>
            </div>
        </section>
        
        <!-- Footer -->
        <footer class="footer">
            <div class="container text-center">
                <div class="cross">
                    <span></span>
                    <span></span>
                </div>
                <h3 class="footer-logo mb-4">
                    <span class="highlight">JESUS</span> CHAT MEMÓRIAS
                </h3>
                <p class="mb-4">Sistema de Indexação para Conversas com IAs</p>
                <div class="footer-links mb-4">
                    <a href="index.html" class="btn btn-outline-light mx-2">Home</a>
                    <a href="search.html" class="btn btn-outline-light mx-2">Buscar</a>
                    <a href="entities.html" class="btn btn-outline-light mx-2">Entidades</a>
                    <a href="graph.html" class="btn btn-outline-light mx-2">Grafo</a>
                    <a href="upload.html" class="btn btn-outline-light mx-2">Upload</a>
                </div>
                <p class="mb-0">&copy; {datetime.datetime.now().year} JESUS CHAT MEMÓRIAS</p>
                <small>Última atualização: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}</small>
            </div>
        </footer>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Dados para os gráficos
            const statsData = {stats_json};
            const wordCloudData = {word_cloud_json};
            
            // Carregar os gráficos quando a página estiver pronta
            document.addEventListener('DOMContentLoaded', function() {
                // Gráfico de modelos
                const modelsCtx = document.getElementById('modelsChart').getContext('2d');
                const modelsLabels = Object.keys(statsData.models);
                const modelsValues = Object.values(statsData.models);
                
                new Chart(modelsCtx, {
                    type: 'pie',
                    data: {
                        labels: modelsLabels,
                        datasets: [{
                            data: modelsValues,
                            backgroundColor: [
                                '#0047AB', '#1E90FF', '#87CEEB', '#4169E1', '#00BFFF', 
                                '#1E3A8A', '#3B82F6', '#7DD3FC', '#0EA5E9', '#0369A1'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'right',
                            },
                            title: {
                                display: false
                            }
                        }
                    }
                });
                
                // Gráfico de tags
                const tagsCtx = document.getElementById('tagsChart').getContext('2d');
                const tagsLabels = Object.keys(statsData.top_tags);
                const tagsValues = Object.values(statsData.top_tags);
                
                new Chart(tagsCtx, {
                    type: 'bar',
                    data: {
                        labels: tagsLabels,
                        datasets: [{
                            label: 'Número de Ocorrências',
                            data: tagsValues,
                            backgroundColor: '#0047AB',
                            borderColor: '#00008B',
                                    """Gera uma visualização HTML da memória."""
        memoria_id = template["metadata"]["id"]
        title = template["metadata"].get("title", "Memória de Chat")
        html_filename = f"{memoria_id}.html"
        html_path = os.path.join(MEMORIES_DIR, html_filename)
        
        # Obter dados para o template HTML
        metadata = template["metadata"]
        summary = template["summary"]
        metrics = template.get("metrics", {})
        messages = template["conversation"].get("messages", [])
        entities = template["semantic_structure"].get("entities", [])
        clusters = template["semantic_structure"].get("topic_clusters", [])
        
        # Criar HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="{metadata.get('language', 'pt-br')}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>JESUS CHAT MEMÓRIAS - {title}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet">
            <style>
                :root {
                    --primary: #0047AB;          /* Azul cobalto */
                    --primary-dark: #00008B;     /* Azul escuro */
                    --secondary: #1E90FF;        /* Azul Dodger */
                    --accent: #87CEEB;           /* Azul céu */
                    --light-blue: #E6F2FF;       /* Azul claro */
                    --dark: #000000;             /* Preto */
                    --dark-gray: #222222;        /* Cinza escuro */
                    --light: #FFFFFF;            /* Branco */
                    --gray: #F0F0F0;             /* Cinza claro */
                }
                
                body {{
                    font-family: 'Montserrat', sans-serif;
                    background-color: var(--gray);
                    color: var(--dark);
                }}
                
                .navbar {{
                    background-color: var(--dark);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }}
                
                .navbar-brand {{
                    font-weight: 800;
                    font-size: 1.4rem;
                    color: var(--light) !important;
                }}
                
                .navbar-brand .highlight {{
                    color: var(--secondary);
                }}
                
                .memory-header {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                    color: var(--light);
                    padding: 3rem 0;
                    position: relative;
                }}
                
                .memory-metadata {{
                    background-color: var(--light);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-top: -2rem;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    margin-bottom: 2rem;
                }}
                
                .metadata-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 0.5rem;
                }}
                
                .metadata-item i {{
                    color: var(--primary);
                    margin-right: 0.5rem;
                    width: 20px;
                    text-align: center;
                }}
                
                .memory-content {{
                    background-color: var(--light);
                    border-radius: 10px;
                    padding: 2rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                    margin-bottom: 2rem;
                }}
                
                .memory-title {{
                    font-weight: 800;
                    color: var(--light);
                    margin-bottom: 1rem;
                }}
                
                .model-badge {{
                    background-color: rgba(255, 255, 255, 0.2);
                    color: var(--light);
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    display: inline-block;
                    margin-bottom: 1rem;
                }}
                
                .section-title {{
                    font-weight: 700;
                    margin-bottom: 1.5rem;
                    color: var(--primary-dark);
                    border-bottom: 2px solid var(--secondary);
                    padding-bottom: 0.5rem;
                }}
                
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 1rem;
                    margin-bottom: 2rem;
                }}
                
                .metric-card {{
                    flex: 1;
                    min-width: 150px;
                    background-color: var(--light-blue);
                    border-radius: 10px;
                    padding: 1rem;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .metric-value {{
                    font-size: 2rem;
                    font-weight: 800;
                    color: var(--primary);
                }}
                
                .metric-label {{
                    font-size: 0.9rem;
                    color: var(--dark-gray);
                    font-weight: 600;
                }}
                
                .tag-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    margin-bottom: 2rem;
                }}
                
                .tag {{
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}
                
                .message {{
                    margin-bottom: 1.5rem;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                
                .message-human {{
                    background-color: var(--light-blue);
                    border-left: 5px solid var(--primary);
                }}
                
                .message-assistant {{
                    background-color: white;
                    border-left: 5px solid var(--secondary);
                }}
                
                .message-header {{
                    padding: 0.8rem 1.5rem;
                    display: flex;
                    justify-content: space-between;
                    background-color: rgba(0, 0, 0, 0.05);
                }}
                
                .message-role {{
                    font-weight: 700;
                }}
                
                .message-time {{
                    font-size: 0.8rem;
                    color: var(--dark-gray);
                }}
                
                .message-content {{
                    padding: 1.5rem;
                    white-space: pre-wrap;
                }}
                
                .message-content pre {{
                    background-color: #f8f9fa;
                    padding: 1rem;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                
                .cluster-card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .cluster-name {{
                    font-weight: 700;
                    margin-bottom: 1rem;
                    color: var(--primary-dark);
                }}
                
                .keyword {{
                    display: inline-block;
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.2rem 0.6rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    margin-right: 0.5rem;
                    margin-bottom: 0.5rem;
                }}
                
                .entity-card {{
                    display: flex;
                    align-items: center;
                    background-color: white;
                    border-radius: 10px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .entity-icon {{
                    width: 40px;
                    height: 40px;
                    background-color: var(--primary);
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 1rem;
                }}
                
                .entity-details {{
                    flex: 1;
                }}
                
                .entity-name {{
                    font-weight: 700;
                    margin-bottom: 0.3rem;
                }}
                
                .entity-type {{
                    font-size: 0.8rem;
                    color: var(--dark-gray);
                }}
                
                .entity-mentions {{
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.2rem 0.6rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}
                
                .summary-box {{
                    background-color: var(--light-blue);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 2rem;
                    border-left: 5px solid var(--primary);
                }}
                
                .insight-item {{
                    margin-bottom: 1rem;
                    padding-left: 1.5rem;
                    position: relative;
                }}
                
                .insight-item::before {{
                    content: "•";
                    color: var(--primary);
                    font-weight: bold;
                    position: absolute;
                    left: 0;
                }}
                
                .action-item {{
                    margin-bottom: 1rem;
                    padding-left: 1.5rem;
                    position: relative;
                }}
                
                .action-item::before {{
                    content: "✓";
                    color: var(--primary);
                    font-weight: bold;
                    position: absolute;
                    left: 0;
                }}
                
                .cross {{
                    width: 40px;
                    height: 40px;
                    margin-bottom: 1rem;
                    position: relative;
                }}
                
                .cross span {{
                    position: absolute;
                    background-color: var(--secondary);
                    border-radius: 2px;
                }}
                
                .cross span:nth-child(1) {{
                    width: 4px;
                    height: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                }}
                
                .cross span:nth-child(2) {{
                    width: 100%;
                    height: 4px;
                    top: 50%;
                    transform: translateY(-50%);
                }}
                
                .footer {{
                    background-color: var(--dark);
                    color: var(--light);
                    padding: 2rem 0;
                    text-align: center;
                }}
                
                .code {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 1rem;
                    overflow-x: auto;
                    font-family: monospace;
                    margin-bottom: 1rem;
                }}
            </style>
        </head>
        <body>
            <!-- Navbar -->
            <nav class="navbar navbar-dark">
                <div class="container">
                    <a class="navbar-brand" href="../index.html">
                        <span class="highlight">JESUS</span> CHAT MEMÓRIAS
                    </a>
                    <div>
                        <a href="../index.html" class="btn btn-outline-light btn-sm">
                            <i class="fas fa-home me-1"></i> Home
                        </a>
                    </div>
                </div>
            </nav>
            
            <!-- Memory Header -->
            <header class="memory-header">
                <div class="container">
                    <div class="model-badge">
                        <i class="fas fa-robot me-1"></i> {metadata.get('model', 'IA')}
                    </div>
                    <h1 class="memory-title">{title}</h1>
                    <p class="lead text-white opacity-75">
                        <i class="far fa-calendar-alt me-1"></i> {self._format_date(metadata.get('timestamp', ''))}
                    </p>
                </div>
            </header>
            
            <!-- Main Content -->
            <div class="container py-4">
                <!-- Metadata -->
                <div class="memory-metadata">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-tag"></i>
                                <span>ID: {memoria_id}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-language"></i>
                                <span>Idioma: {metadata.get('language', 'Não especificado')}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-comment-dots"></i>
                                <span>Mensagens: {len(messages)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-clock"></i>
                                <span>Duração: {metadata.get('duration_seconds', 0)} segundos</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-calculator"></i>
                                <span>Tokens: {metadata.get('total_tokens', 0)}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-server"></i>
                                <span>Plataforma: {metadata.get('platform', 'Não especificada')}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Summary -->
                <div class="memory-content">
                    <h2 class="section-title">Resumo</h2>
                    <div class="summary-box">
                        <p class="mb-0">{summary.get('brief', 'Não há resumo disponível.')}</p>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h3 class="section-title h5">Insights Principais</h3>
                            <div class="insights-container">
                                {self._generate_insights_html(summary.get('key_insights', []))}
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h3 class="section-title h5">Ações Recomendadas</h3>
                            <div class="actions-container">
                                {self._generate_actions_html(summary.get('action_items', []))}
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Metrics -->
                <div class="memory-content">
                    <h2 class="section-title">Métricas</h2>
                    <div class="metrics-container">
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('problem_resolution_score', 0))}</div>
                            <div class="metric-label">Resolução</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('technical_accuracy', 0))}</div>
                            <div class="metric-label">Precisão</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('response_completeness', 0))}</div>
                            <div class="metric-label">Completude</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('chat_efficiency', 0))}</div>
                            <div class="metric-label">Eficiência</div>
                        </div>
                    </div>
                </div>
                
                <!-- Tags -->
                <div class="memory-content">
                    <h2 class="section-title">Tags</h2>
                    <div class="tag-container">
                        {self._generate_tags_html(metadata.get('tags', []))}
                    </div>
                </div>
                
                <!-- Clusters -->
                <div class="memory-content">
                    <h2 class="section-title">Clusters Temáticos</h2>
                    <div class="clusters-container">
                        {self._generate_clusters_html(clusters)}
                    </div>
                </div>
                
                <!-- Entities -->
                <div class="memory-content">
                    <h2 class="section-title">Entidades</h2>
                    <div class="entities-container">
                        {self._generate_entities_html(entities)}
                    </div>
                </div>
                
                <!-- Chat -->
                <div class="memory-content">
                    <h2 class="section-title">Conversa</h2>
                    <div class="conversation-container">
                        {self._generate_messages_html(messages)}
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <footer class="footer">
                <div class="container">
                    <div class="cross">
                        <span></span>
                        <span></span>
                    </div>
                    <p class="mb-0">JESUS CHAT MEMÓRIAS &copy; {datetime.datetime.now().year}</p>
                    <small>Memória gerada em {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}</small>
                </div>
            </footer>
        </body>
        </html>
        """
        
        # Salvar arquivo HTML
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _format_date(self, timestamp: str) -> str:
        """Formata timestamp para exibição."""
        if not timestamp:
            return "Data não especificada"
        
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%d/%m/%Y %H:%M')
        except (ValueError, TypeError):
            return timestamp
    
    def _format_percentage(self, value: float) -> str:
        """Formata valor de métrica como porcentagem."""
        if not value:
            return "0%"
        
        return f"{int(value * 100)}%"
    
    def _generate_insights_html(self, insights: List[str]) -> str:
        """Gera HTML para insights."""
        if not insights:
            return "<p>Nenhum insight disponível.</p>"
        
        html = ""
        for insight in insights:
            html += f'<div class="insight-item">{insight}</div>'
        
        return html
    
    def _generate_actions_html(self, actions: List[str]) -> str:
        """Gera HTML para ações recomendadas."""
        if not actions:
            return "<p>Nenhuma ação recomendada.</p>"
        
        html = ""
        for action in actions:
            html += f'<div class="action-item">{action}</div>'
        
        return html
    
    def _generate_tags_html(self, tags: List[str]) -> str:
        """Gera HTML para tags."""
        if not tags:
            return "<span class='tag'>sem-tags</span>"
        
        html = ""
        for tag in tags:
            html += f'<span class="tag">{tag}</span>'
        
        return html
    
    def _generate_clusters_html(self, clusters: List[Dict]) -> str:
        """Gera HTML para clusters temáticos."""
        if not clusters:
            return "<p>Nenhum cluster temático identificado.</p>"
        
        html = ""
        for cluster in clusters:
            keywords_html = ""
            for keyword in cluster.get("keywords", []):
                keywords_html += f'<span class="keyword">{keyword}</span>'
            
            importance = cluster.get("importance", 0)
            importance_percent = self._format_percentage(importance)
            
            html += f"""
            <div class="cluster-card">
                <div class="d-flex justify-content-between align-items-center">
                    <h3 class="cluster-name">{cluster.get("name", "Sem nome")}</h3>
                    <span class="entity-mentions">Relevância: {importance_percent}</span>
                </div>
                <div class="keywords-container">
                    {keywords_html}
                </div>
            </div>
            """
        
        return html
    
    def _generate_entities_html(self, entities: List[Dict]) -> str:
        """Gera HTML para entidades."""
        if not entities:
            return "<p>Nenhuma entidade identificada.</p>"
        
        html = ""
        for entity in entities:
            entity_type = entity.get("type", "conceito")
            icon = self._get_entity_icon(entity_type)
            
            html += f"""
            <div class="entity-card">
                <div class="entity-icon">
                    <i class="{icon}"></i>
                </div>
                <div class="entity-details">
                    <div class="entity-name">{entity.get("name", "Sem nome")}</div>
                    <div class="entity-type">Tipo: {entity_type}</div>
                </div>
                <span class="entity-mentions">{entity.get("mentions", 0)} menções</span>
            </div>
            """
        
        return html
    
    def _get_entity_icon(self, entity_type: str) -> str:
        """Retorna ícone baseado no tipo de entidade."""
        icons = {
            "technology": "fas fa-microchip",
            "language": "fas fa-code",
            "concept": "fas fa-lightbulb",
            "person": "fas fa-user",
            "organization": "fas fa-building",
            "location": "fas fa-map-marker-alt",
            "protocol": "fas fa-shield-alt",
            "application": "fas fa-cube",
            "framework": "fas fa-layer-group",
            "architecture": "fas fa-sitemap"
        }
        
        return icons.get(entity_type.lower(), "fas fa-tag")
    
    def _generate_messages_html(self, messages: List[Dict]) -> str:
        """Gera HTML para mensagens."""
        if not messages:
            return "<p>Nenhuma mensagem disponível.</p>"
        
        html = ""
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = self._format_date(message.get("timestamp", ""))
            
            # Formatação de código em blocos de código
            content = self._format_code_blocks(content)
            
            message_class = "message-human" if role == "human" else "message-assistant"
            role_display = "Usuário" if role == "human" else "Assistente"
            
            html += f"""
            <div class="message {message_class}">
                <div class="message-header">
                    <div class="message-role">{role_display}</div>
                    <div class="message-time">{timestamp}</div>
                </div>
                <div class="message-content">
                    {content}
                </div>
            </div>
            """
        
        return html
    
    def _format_code_blocks(self, content: str) -> str:
        """Formata blocos de código no conteúdo."""
        # Padrão para corresponder a blocos de código markdown
        pattern = r'```(.*?)\n(.*?)```'
        
        def replace_code_block(match):
            language = match.group(1).strip()
            code = match.group(2)
            return f'<pre><code class="language-{language}">{code}</code></pre>'
        
        # Substituir blocos de código
        formatted_content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)
        
        return formatted_content
    
    def search_chats(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats com base em uma consulta textual.
        
        Args:
            query: Texto para busca
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        # Busca por título, resumo e tags
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        WHERE 
            m.title LIKE ? OR 
            m.summary_brief LIKE ? OR 
            m.tags LIKE ?
        ORDER BY m.timestamp DESC
        LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def search_by_entity(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats que mencionam uma entidade específica.
        
        Args:
            entity_name: Nome da entidade
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        JOIN entities e ON m.id = e.memoria_id
        WHERE e.name LIKE ?
        ORDER BY e.mentions DESC
        LIMIT ?
        """, (f"%{entity_name}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def search_by_cluster(self, cluster_topic: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats relacionados a um cluster/tópico específico.
        
        Args:
            cluster_topic: Tópico ou palavra-chave do cluster
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        JOIN clusters c ON m.id = c.memoria_id
        WHERE c.name LIKE ? OR c.keywords LIKE ?
        ORDER BY c.importance DESC
        LIMIT ?
        """, (f"%{cluster_topic}%", f"%{cluster_topic}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def export_to_csv(self, output_path: str = "jesus_chat_memorias_export.csv") -> str:
        """
        Exporta o banco de dados de memórias para um arquivo CSV.
        
        Args:
            output_path: Caminho para salvar o arquivo CSV
            
        Returns:
            Caminho do arquivo CSV exportado
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT 
            m.id, m.timestamp, m.title, m.model, m.language, m.import os
import json
import uuid
import hashlib
import datetime
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import numpy as np

# Configurações
DB_PATH = "db/jesus_chat_memorias.sqlite"
MEMORIES_DIR = "memorias"
TEMPLATES_DIR = "templates"
HTML_TEMPLATE_PATH = "app/templates/memory_template.html"

class MemoriaChatProcessor:
    """
    Processador para memórias de chat com IAs.
    Processa templates JSON e cria representações em HTML e banco de dados.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """Inicializa o processador."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(MEMORIES_DIR, exist_ok=True)
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        
        self.db_path = db_path
        self.conn = self._init_database()
    
    def _init_database(self) -> sqlite3.Connection:
        """Inicializa o banco de dados SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Criar tabela memorias
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memorias (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            title TEXT,
            model TEXT,
            language TEXT,
            summary_brief TEXT,
            problem_resolution_score REAL,
            response_completeness REAL,
            technical_accuracy REAL,
            chat_efficiency REAL,
            tags TEXT,
            filepath_csv TEXT,
            filepath_html TEXT,
            created_at TEXT
        )
        """)
        
        # Criar tabela clusters
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            name TEXT,
            keywords TEXT,
            importance REAL,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela entities
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            mentions INTEGER,
            related_clusters TEXT,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela knowledge_graph (nós)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            type TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela edges (arestas)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY,
            source TEXT,
            target TEXT,
            relationship TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (source) REFERENCES graph_nodes (id),
            FOREIGN KEY (target) REFERENCES graph_nodes (id),
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela messages
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            tokens INTEGER,
            clusters TEXT,
            sentiment TEXT,
            intent TEXT,
            key_points TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id ON clusters (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_entities ON entities (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_messages ON messages (memoria_id)")
        
        conn.commit()
        return conn
    
    def process_template(self, template_path: str) -> str:
        """
        Processa um template JSON de chat e armazena no banco de dados.
        
        Args:
            template_path: Caminho para o arquivo JSON do template
            
        Returns:
            ID da memória processada
        """
        # Carregar template
        with open(template_path, 'r', encoding='utf-8') as f:
            try:
                template = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Erro ao decodificar o template JSON: {e}")
        
        # Validar template
        self._validate_template(template)
        
        # Obter ID da memória
        memoria_id = template["metadata"]["id"]
        if not memoria_id:
            memoria_id = f"chat_{uuid.uuid4()}"
            template["metadata"]["id"] = memoria_id
        
        # Processar e salvar no banco de dados
        self._save_to_database(template)
        
        # Gerar arquivos
        csv_path = self._generate_csv_export(template)
        html_path = self._generate_html_view(template)
        
        # Atualizar caminhos de arquivo
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memorias SET filepath_csv = ?, filepath_html = ? WHERE id = ?",
            (csv_path, html_path, memoria_id)
        )
        self.conn.commit()
        
        print(f"✅ Memória processada e salva com sucesso: {memoria_id}")
        return memoria_id
    
    def _validate_template(self, template: Dict) -> bool:
        """Valida a estrutura do template."""
        required_sections = ["metadata", "semantic_structure", "conversation", "summary"]
        for section in required_sections:
            if section not in template:
                raise ValueError(f"Seção obrigatória ausente no template: {section}")
        
        # Validar metadata
        required_metadata = ["id", "title", "timestamp"]
        for field in required_metadata:
            if field not in template["metadata"]:
                template["metadata"][field] = "" if field != "timestamp" else datetime.datetime.now().isoformat()
        
        return True
    
    def _save_to_database(self, template: Dict) -> None:
        """Salva o template processado no banco de dados."""
        cursor = self.conn.cursor()
        memoria_id = template["metadata"]["id"]
        
        # Salvar memória principal
        tags = ",".join(template["metadata"].get("tags", []))
        metrics = template.get("metrics", {})
        
        cursor.execute("""
        INSERT OR REPLACE INTO memorias (
            id, timestamp, title, model, language, summary_brief, 
            problem_resolution_score, response_completeness, technical_accuracy, 
            chat_efficiency, tags, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memoria_id,
            template["metadata"].get("timestamp", ""),
            template["metadata"].get("title", ""),
            template["metadata"].get("model", ""),
            template["metadata"].get("language", ""),
            template["summary"].get("brief", ""),
            metrics.get("problem_resolution_score", 0),
            metrics.get("response_completeness", 0),
            metrics.get("technical_accuracy", 0),
            metrics.get("chat_efficiency", 0),
            tags,
            datetime.datetime.now().isoformat()
        ))
        
        # Salvar clusters
        for cluster in template["semantic_structure"].get("topic_clusters", []):
            cluster_id = cluster.get("id", f"cluster_{uuid.uuid4()}")
            keywords = ",".join(cluster.get("keywords", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO clusters (id, memoria_id, name, keywords, importance)
            VALUES (?, ?, ?, ?, ?)
            """, (
                cluster_id,
                memoria_id,
                cluster.get("name", ""),
                keywords,
                cluster.get("importance", 0)
            ))
        
        # Salvar entidades
        for entity in template["semantic_structure"].get("entities", []):
            entity_id = f"entity_{hashlib.md5(entity.get('name', '').encode()).hexdigest()[:8]}"
            related_clusters = ",".join(entity.get("related_clusters", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO entities (id, name, type, mentions, related_clusters, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entity_id,
                entity.get("name", ""),
                entity.get("type", ""),
                entity.get("mentions", 0),
                related_clusters,
                memoria_id
            ))
        
        # Salvar nós do grafo
        for node in template["semantic_structure"].get("knowledge_graph", {}).get("nodes", []):
            cursor.execute("""
            INSERT OR REPLACE INTO graph_nodes (id, label, type, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?)
            """, (
                node.get("id", ""),
                node.get("label", ""),
                node.get("type", ""),
                node.get("weight", 0),
                memoria_id
            ))
        
        # Salvar arestas do grafo
        for idx, edge in enumerate(template["semantic_structure"].get("knowledge_graph", {}).get("edges", [])):
            edge_id = f"edge_{idx}_{memoria_id}"
            cursor.execute("""
            INSERT OR REPLACE INTO graph_edges (id, source, target, relationship, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                edge.get("source", ""),
                edge.get("target", ""),
                edge.get("relationship", ""),
                edge.get("weight", 0),
                memoria_id
            ))
        
        # Salvar mensagens
        for message in template["conversation"].get("messages", []):
            message_id = message.get("id", f"msg_{uuid.uuid4()}")
            clusters = ",".join(message.get("clusters", []))
            key_points = ",".join(message.get("key_points", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO messages (
                id, memoria_id, role, content, timestamp, tokens, 
                clusters, sentiment, intent, key_points
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                memoria_id,
                message.get("role", ""),
                message.get("content", ""),
                message.get("timestamp", ""),
                message.get("tokens", 0),
                clusters,
                message.get("sentiment", ""),
                message.get("intent", ""),
                key_points
            ))
        
        self.conn.commit()
    
    def _generate_csv_export(self, template: Dict) -> str:
        """Gera um arquivo CSV da memória."""
        memoria_id = template["metadata"]["id"]
        csv_filename = f"{memoria_id}.csv"
        csv_path = os.path.join(MEMORIES_DIR, csv_filename)
        
        # Criar DataFrame para mensagens
        messages = []
        for msg in template["conversation"].get("messages", []):
            messages.append({
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
                "tokens": msg.get("tokens", 0),
                "sentiment": msg.get("sentiment", ""),
                "intent": msg.get("intent", ""),
                "key_points": ", ".join(msg.get("key_points", []))
            })
        
        df = pd.DataFrame(messages)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return csv_path
    
    def _generate_html_view(self, template: Dict) -> str:
        """Gera uma visualização HTML da memória."""
        memoria_id = template["metadata"]["id"]
        title = template["metadata"].get
        borderColor: '#00008B',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                display: false
                            },
                            title: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                
                // Nuvem de palavras
                createWordCloud(wordCloudData.words);
            });
            
            // Função para criar nuvem de palavras
            function createWordCloud(words) {
                const width = document.getElementById('wordCloud').clientWidth;
                const height = 400;
                
                // Configurar layout
                const layout = d3.layout.cloud()
                    .size([width, height])
                    .words(words)
                    .padding(5)
                    .rotate(() => ~~(Math.random() * 2) * 90)
                    .font("Impact")
                    .fontSize(d => Math.sqrt(d.value) * 5)
                    .on("end", draw);
                
                layout.start();
                
                // Função para desenhar
                function draw(words) {
                    d3.select("#wordCloud").html("");
                    
                    const svg = d3.select("#wordCloud").append("svg")
                        .attr("width", width)
                        .attr("height", height)
                        .append("g")
                        .attr("transform", `translate(${width/2},${height/2})`);
                    
                    svg.selectAll("text")
                        .data(words)
                        .enter().append("text")
                        .style("font-size", d => `${d.size}px`)
                        .style("font-family", "Impact")
                        .style("fill", (d, i) => d3.interpolateBlues(i / 20))
                        .attr("text-anchor", "middle")
                        .attr("transform", d => `translate(${d.x},${d.y})rotate(${d.rotate})`)
                        .text(d => d.text);
                }
            }
            
            // Busca
            const searchInput = document.querySelector('.search-input');
            const searchButton = document.querySelector('.search-button');
            
            searchButton.addEventListener('click', function() {
                const query = searchInput.value.trim();
                if (query) {
                    window.location.href = `search.html?q=${encodeURIComponent(query)}`;
                }
            });
            
            searchInput.addEventListener('keyup', function(e) {
                if (e.key === 'Enter') {
                    searchButton.click();
                }
            });
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path

def _generate_memory_cards_html(processor, memories):
    """Gera HTML para cards de memórias."""
    if not memories:
        return "<p class='text-center'>Nenhuma memória encontrada.</p>"
    
    html = ""
    for memory in memories:
        # Extrair dados
        memory_id = memory[0]
        title = memory[1]
        model = memory[2]
        timestamp = memory[3]
        summary = memory[4]
        tags = memory[5].split(',') if memory[5] else []
        html_path = memory[6]
        
        # Formatar data
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date_formatted = dt.strftime('%d/%m/%Y')
        except (ValueError, TypeError, AttributeError):
            date_formatted = timestamp
        
        # Gerar HTML para tags
        tags_html = ""
        for tag in tags[:3]:  # Limitar a 3 tags
            if tag:
                tags_html += f'<span class="memory-tag">{tag}</span>'
        
        if len(tags) > 3:
            tags_html += f'<span class="memory-tag">+{len(tags)-3}</span>'
        
        # Gerar card
        html += f"""
        <div class="col-lg-4 col-md-6">
            <div class="memory-card">
                <div class="memory-header">
                    <div class="memory-model">
                        <i class="fas fa-robot me-1"></i> {model or "IA"}
                    </div>
                    <h3 class="memory-title">{title}</h3>
                    <div class="memory-date">
                        <i class="far fa-calendar-alt me-1"></i> {date_formatted}
                    </div>
                </div>
                <div class="memory-body">
                    <div class="memory-summary">
                        {summary or "Sem resumo disponível."}
                    </div>
                    <div class="memory-tags">
                        {tags_html}
                    </div>
                </div>
                <div class="memory-footer">
                    <a href="{html_path}" class="btn btn-view w-100">
                        <i class="fas fa-eye me-1"></i> Ver Detalhes
                    </a>
                </div>
            </div>
        </div>
        """
    
    return html


def main():
    """Função principal do processador."""
    # Inicializar processador
    processor = MemoriaChatProcessor()
    
    # Verificar se existe diretório de templates
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
        print(f"Diretório {TEMPLATES_DIR} criado. Adicione arquivos JSON de template aqui.")
    
    # Processar templates existentes
    if os.listdir(TEMPLATES_DIR):
        print(f"Processando templates em {TEMPLATES_DIR}...")
        processed_ids = process_directory(processor, TEMPLATES_DIR)
        print(f"Total de {len(processed_ids)} templates processados.")
    else:
        print(f"Nenhum template encontrado em {TEMPLATES_DIR}.")
    
    # Gerar página inicial
    index_path = generate_index_html(processor)
    print(f"Página inicial gerada em {index_path}")
    
    # Gerar páginas adicionais (busca, entidades, grafo)
    # TODO: Implementar geração de páginas adicionais
    
    # Exportar CSV geral
    csv_path = processor.export_to_csv()
    print(f"Banco de dados exportado para {csv_path}")
    
    # Fechar conexão
    processor.close()
    
    print("\n✅ Processamento concluído! Sistema Jesus Chat Memórias está pronto.")
    print(f"Acesse a página inicial em: {os.path.abspath(index_path)}")


if __name__ == "__main__":
    main()
            m.id, m.timestamp, m.title, m.model, m.language, m.summary_brief, 
            m.problem_resolution_score, m.response_completeness, m.technical_accuracy, 
            m.chat_efficiency, m.tags, m.filepath_csv, m.filepath_html, m.created_at
        FROM memorias m
        """)
        
        results = cursor.fetchall()
        
        # Criar DataFrame para exportação
        columns = [
            "id", "timestamp", "title", "model", "language", "summary_brief",
            "problem_resolution_score", "response_completeness", "technical_accuracy",
            "chat_efficiency", "tags", "filepath_csv", "filepath_html", "created_at"
        ]
        
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path
    
    def generate_graph_data(self) -> Dict:
        """
        Gera dados para visualização do grafo de conhecimento.
        
        Returns:
            Dicionário com nós e arestas para visualização
        """
        cursor = self.conn.cursor()
        
        # Buscar todos os nós
        cursor.execute("""
        SELECT id, label, type, weight, memoria_id FROM graph_nodes
        """)
        nodes_data = cursor.fetchall()
        
        # Buscar todas as arestas
        cursor.execute("""
        SELECT source, target, relationship, weight FROM graph_edges
        """)
        edges_data = cursor.fetchall()
        
        # Formatar para visualização
        nodes = []
        for node in nodes_data:
            nodes.append({
                "id": node[0],
                "label": node[1],
                "type": node[2],
                "weight": node[3],
                "memoria_id": node[4]
            })
        
        edges = []
        for edge in edges_data:
            edges.append({
                "source": edge[0],
                "target": edge[1],
                "relationship": edge[2],
                "weight": edge[3]
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def generate_word_cloud_data(self) -> Dict:
        """
        Gera dados para nuvem de palavras baseada em entidades e keywords.
        
        Returns:
            Dicionário com palavras e suas contagens
        """
        cursor = self.conn.cursor()
        
        # Buscar entidades e suas menções
        cursor.execute("""
        SELECT name, SUM(mentions) as total_mentions FROM entities
        GROUP BY name
        ORDER BY total_mentions DESC
        LIMIT 100
        """)
        entity_data = cursor.fetchall()
        
        # Buscar keywords dos clusters
        cursor.execute("""
        SELECT keywords FROM clusters
        """)
        keywords_data = cursor.fetchall()
        
        # Processar entidades
        word_counts = {}
        for entity in entity_data:
            word_counts[entity[0]] = int(entity[1])
        
        # Processar keywords
        for keywords_row in keywords_data:
            if keywords_row[0]:
                keywords = keywords_row[0].split(',')
                for keyword in keywords:
                    if keyword:
                        word_counts[keyword] = word_counts.get(keyword, 0) + 5  # Peso adicional para keywords
        
        # Formatar resultado
        word_cloud_data = [{"text": word, "value": count} for word, count in word_counts.items()]
        
        return {
            "words": word_cloud_data
        }
    
    def get_stats(self) -> Dict:
        """
        Obtém estatísticas gerais do banco de dados.
        
        Returns:
            Dicionário com estatísticas
        """
        cursor = self.conn.cursor()
        
        # Total de memórias
        cursor.execute("SELECT COUNT(*) FROM memorias")
        total_memorias = cursor.fetchone()[0]
        
        # Total de entidades
        cursor.execute("SELECT COUNT(*) FROM entities")
        total_entities = cursor.fetchone()[0]
        
        # Total de clusters
        cursor.execute("SELECT COUNT(*) FROM clusters")
        total_clusters = cursor.fetchone()[0]
        
        # Total de mensagens
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        # Modelos utilizados
        cursor.execute("""
        SELECT model, COUNT(*) as count FROM memorias
        GROUP BY model
        ORDER BY count DESC
        """)
        models_data = cursor.fetchall()
        models = {model: count for model, count in models_data}
        
        # Tags mais comuns
        cursor.execute("""
        SELECT tags FROM memorias
        """)
        tags_rows = cursor.fetchall()
        
        tag_counts = {}
        for row in tags_rows:
            if row[0]:
                tags = row[0].split(',')
                for tag in tags:
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            "total_memorias": total_memorias,
            "total_entities": total_entities,
            "total_clusters": total_clusters,
            "total_messages": total_messages,
            "models": models,
            "top_tags": top_tags,
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self.conn:
            self.conn.close()


def process_directory(processor: MemoriaChatProcessor, directory: str) -> List[str]:
    """
    Processa todos os arquivos JSON em um diretório.
    
    Args:
        processor: Instância do processador
        directory: Caminho do diretório com templates JSON
        
    Returns:
        Lista de IDs das memórias processadas
    """
    processed_ids = []
    
    # Listar arquivos JSON no diretório
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            memoria_id = processor.process_template(file_path)
            processed_ids.append(memoria_id)
            print(f"Processado: {json_file} → ID: {memoria_id}")
        except Exception as e:
            print(f"Erro ao processar {json_file}: {e}")
    
    return processed_ids


def generate_index_html(processor: MemoriaChatProcessor, output_path: str = "index.html") -> str:
    """
    Gera página HTML inicial com dashboard.
    
    Args:
        processor: Instância do processador
        output_path: Caminho para salvar o arquivo HTML
        
    Returns:
        Caminho do arquivo HTML gerado
    """
    # Obter estatísticas
    stats = processor.get_stats()
    
    # Obter dados para nuvem de palavras
    word_cloud_data = processor.generate_word_cloud_data()
    
    # Converter para JSON para uso no JavaScript
    stats_json = json.dumps(stats)
    word_cloud_json = json.dumps(word_cloud_data)
    
    # Cursor para obter memórias recentes
    cursor = processor.conn.cursor()
    cursor.execute("""
    SELECT id, title, model, timestamp, summary_brief, tags, filepath_html
    FROM memorias
    ORDER BY timestamp DESC
    LIMIT 6
    """)
    
    recent_memories = cursor.fetchall()
    
    # Criar HTML da página inicial
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JESUS CHAT MEMÓRIAS</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        <script src="https://cdn.jsdelivr.net/npm/d3-cloud@1.2.5/build/d3.layout.cloud.min.js"></script>
        <style>
            :root {
                --primary: #0047AB;          /* Azul cobalto */
                --primary-dark: #00008B;     /* Azul escuro */
                --secondary: #1E90FF;        /* Azul Dodger */
                --accent: #87CEEB;           /* Azul céu */
                --light-blue: #E6F2FF;       /* Azul claro */
                --dark: #000000;             /* Preto */
                --dark-gray: #222222;        /* Cinza escuro */
                --light: #FFFFFF;            /* Branco */
                --gray: #F0F0F0;             /* Cinza claro */
            }
            
            body {
                font-family: 'Montserrat', sans-serif;
                background-color: var(--gray);
                color: var(--dark);
            }
            
            /* Navbar */
            .navbar {
                background-color: var(--dark);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            
            .navbar-brand {
                font-weight: 800;
                font-size: 1.4rem;
                color: var(--light) !important;
            }
            
            .navbar-brand .highlight {
                color: var(--secondary);
            }
            
            .nav-link {
                color: var(--light) !important;
                font-weight: 600;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                margin: 0 0.2rem;
                border-radius: 4px;
            }
            
            .nav-link:hover, 
            .nav-link.active {
                background-color: var(--primary);
                color: var(--light) !important;
            }
            
            /* Hero Section */
            .hero-section {
                background: linear-gradient(135deg, var(--dark) 0%, var(--primary-dark) 100%);
                color: var(--light);
                padding: 5rem 0;
                position: relative;
                overflow: hidden;
            }
            
            .hero-title {
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 1.5rem;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            }
            
            .hero-subtitle {
                font-size: 1.5rem;
                margin-bottom: 2rem;
                opacity: 0.9;
            }
            
            .cross-icon {
                font-size: 5rem;
                color: var(--secondary);
                margin-bottom: 2rem;
                text-shadow: 0 0 20px rgba(30, 144, 255, 0.7);
            }
            
            /* Stats Cards */
            .stats-section {
                margin-top: -60px;
                z-index: 10;
                position: relative;
                padding-bottom: 3rem;
            }
            
            .stats-card {
                background: var(--light);
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                text-align: center;
                transition: all 0.3s ease;
                border-top: 5px solid var(--primary);
                margin-bottom: 1.5rem;
                height: 100%;
            }
            
            .stats-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
            }
            
            .stats-icon {
                height: 80px;
                width: 80px;
                background: var(--light-blue);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1.5rem;
                color: var(--primary);
                font-size: 2rem;
                box-shadow: 0 5px 15px rgba(0, 71, 171, 0.2);
            }
            
            .stats-value {
                font-size: 2.5rem;
                font-weight: 800;
                color: var(--primary);
                margin-bottom: 0.5rem;
            }
            
            .stats-label {
                font-size: 1.1rem;
                color: var(--dark-gray);
                font-weight: 600;
            }
            
            /* Content Sections */
            .content-section {
                padding: 5rem 0;
            }
            
            .content-section.bg-light {
                background-color: var(--light);
            }
            
            .section-title {
                font-size: 2rem;
                font-weight: 800;
                text-align: center;
                margin-bottom: 1rem;
                color: var(--dark);
            }
            
            .section-subtitle {
                font-size: 1.2rem;
                text-align: center;
                margin-bottom: 3rem;
                color: var(--primary);
            }
            
            /* Memory Cards */
            .memory-card {
                background: var(--light);
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
                height: 100%;
                display: flex;
                flex-direction: column;
                margin-bottom: 2rem;
            }
            
            .memory-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
            }
            
            .memory-header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: var(--light);
                padding: 1.5rem;
                border-bottom: 3px solid var(--secondary);
            }
            
            .memory-model {
                display: inline-block;
                background: rgba(255, 255, 255, 0.2);
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .memory-title {
                font-size: 1.3rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .memory-date {
                font-size: 0.9rem;
                opacity: 0.8;
            }
            
            .memory-body {
                padding: 1.5rem;
                flex-grow: 1;
            }
            
            .memory-summary {
                color: var(--dark-gray);
                margin-bottom: 1.5rem;
                font-size: 0.95rem;
                line-height: 1.6;
                display: -webkit-box;
                -webkit-line-clamp: 4;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            
            .memory-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .memory-tag {
                display: inline-block;
                padding: 0.3rem 0.8rem;
                background: var(--light-blue);
                border-radius: 20px;
                font-size: 0.8rem;
                color: var(--primary-dark);
                font-weight: 600;
            }
            
            .memory-footer {
                padding: 1.5rem;
                border-top: 1px solid var(--gray);
            }
            
            .btn-view {
                background-color: var(--primary);
                color: var(--light);
                border: none;
                border-radius: 30px;
                padding: 0.5rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .btn-view:hover {
                background-color: var(--primary-dark);
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            /* Charts */
            .chart-container {
                background: var(--light);
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
                height: 100%;
            }
            
            .word-cloud-container {
                height: 400px;
                position: relative;
            }
            
            /* Footer */
            .footer {
                background-color: var(--dark);
                color: var(--light);
                padding: 4rem 0 2rem;
            }
            
            .footer-logo {
                font-size: 1.8rem;
                font-weight: 800;
                margin-bottom: 1rem;
            }
            
            .footer-logo .highlight {
                color: var(--secondary);
            }
            
            .cross {
                position: relative;
                width: 60px;
                height: 60px;
                margin: 0 auto 2rem;
            }
            
            .cross span {
                position: absolute;
                background-color: var(--secondary);
                border-radius: 4px;
            }
            
            .cross span:nth-child(1) {
                width: 4px;
                height: 100%;
                left: 50%;
                transform: translateX(-50%);
                animation: pulse 2s infinite;
            }
            
            .cross span:nth-child(2) {
                width: 100%;
                height: 4px;
                top: 50%;
                transform: translateY(-50%);
                animation: pulse 2s infinite 0.3s;
            }
            
            @keyframes pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(30, 144, 255, 0.7);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(30, 144, 255, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(30, 144, 255, 0);
                }
            }
            
            .search-container {
                background: var(--light);
                border-radius: 30px;
                overflow: hidden;
                display: flex;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            
            .search-input {
                flex: 1;
                border: none;
                padding: 1rem 1.5rem;
                font-size: 1rem;
            }
            
            .search-input:focus {
                outline: none;
            }
            
            .search-button {
                background: var(--primary);
                color: white;
                border: none;
                padding: 0 1.5rem;
                font-size: 1.2rem;
            }
            
            .search-button:hover {
                background: var(--primary-dark);
            }
        </style>
    </head>
    <body>
        <!-- Navbar -->
        <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
            <div class="container">
                <a class="navbar-brand" href="index.html">
                    <span class="highlight">JESUS</span> CHAT MEMÓRIAS
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav ms-auto">
                        <li class="nav-item">
                            <a class="nav-link active" href="index.html">
                                <i class="fas fa-home me-1"></i> Home
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="search.html">
                                <i class="fas fa-search me-1"></i> Buscar
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="entities.html">
                                <i class="fas fa-tags me-1"></i> Entidades
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="graph.html">
                                <i class="fas fa-project-diagram me-1"></i> Grafo
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="upload.html">
                                <i class="fas fa-upload me-1"></i> Upload
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <!-- Hero Section -->
        <section class="hero-section">
            <div class="container text-center">
                <div class="cross">
                    <span></span>
                    <span></span>
                </div>
                <h1 class="hero-title">JESUS CHAT MEMÓRIAS</h1>
                <p class="hero-subtitle">Sistema de Indexação para Conversas com IAs</p>
                <div class="search-container mx-auto" style="max-width: 600px;">
                    <input type="text" class="search-input" placeholder="Buscar por assunto, entidade ou tag...">
                    <button class="search-button">
                        <i class="fas fa-search"></i>
                    </button>
                </div>
            </div>
        </section>
        
        <!-- Stats Section -->
        <section class="stats-section">
            <div class="container">
                <div class="row">
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-comments"></i>
                            </div>
                            <div class="stats-value" id="memoriasCount">{stats['total_memorias']}</div>
                            <div class="stats-label">Memórias</div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-sitemap"></i>
                            </div>
                            <div class="stats-value" id="clustersCount">{stats['total_clusters']}</div>
                            <div class="stats-label">Clusters</div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-tag"></i>
                            </div>
                            <div class="stats-value" id="entitiesCount">{stats['total_entities']}</div>
                            <div class="stats-label">Entidades</div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="stats-card">
                            <div class="stats-icon">
                                <i class="fas fa-comment-dots"></i>
                            </div>
                            <div class="stats-value" id="messagesCount">{stats['total_messages']}</div>
                            <div class="stats-label">Mensagens</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Charts Section -->
        <section class="content-section bg-light">
            <div class="container">
                <h2 class="section-title">Análise de Dados</h2>
                <p class="section-subtitle">Visualizações e métricas das suas conversas com IAs</p>
                
                <div class="row">
                    <div class="col-lg-6 mb-4">
                        <div class="chart-container">
                            <h3 class="h5 mb-4">Distribuição de Modelos</h3>
                            <canvas id="modelsChart"></canvas>
                        </div>
                    </div>
                    <div class="col-lg-6 mb-4">
                        <div class="chart-container">
                            <h3 class="h5 mb-4">Tags Mais Comuns</h3>
                            <canvas id="tagsChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12 mb-4">
                        <div class="chart-container">
                            <h3 class="h5 mb-4">Nuvem de Palavras</h3>
                            <div class="word-cloud-container" id="wordCloud"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Recent Memories Section -->
        <section class="content-section">
            <div class="container">
                <h2 class="section-title">Memórias Recentes</h2>
                <p class="section-subtitle">Últimas conversas processadas</p>
                
                <div class="row">
                    {self._generate_memory_cards_html(recent_memories)}
                </div>
                
                <div class="text-center mt-4">
                    <a href="search.html" class="btn btn-primary btn-lg">
                        <i class="fas fa-search me-2"></i>Ver Todas as Memórias
                    </a>
                </div>
            </div>
        </section>
        
        <!-- Footer -->
        <footer class="footer">
            <div class="container text-center">
                <div class="cross">
                    <span></span>
                    <span></span>
                </div>
                <h3 class="footer-logo mb-4">
                    <span class="highlight">JESUS</span> CHAT MEMÓRIAS
                </h3>
                <p class="mb-4">Sistema de Indexação para Conversas com IAs</p>
                <div class="footer-links mb-4">
                    <a href="index.html" class="btn btn-outline-light mx-2">Home</a>
                    <a href="search.html" class="btn btn-outline-light mx-2">Buscar</a>
                    <a href="entities.html" class="btn btn-outline-light mx-2">Entidades</a>
                    <a href="graph.html" class="btn btn-outline-light mx-2">Grafo</a>
                    <a href="upload.html" class="btn btn-outline-light mx-2">Upload</a>
                </div>
                <p class="mb-0">&copy; {datetime.datetime.now().year} JESUS CHAT MEMÓRIAS</p>
                <small>Última atualização: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}</small>
            </div>
        </footer>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Dados para os gráficos
            const statsData = {stats_json};
            const wordCloudData = {word_cloud_json};
            
            // Carregar os gráficos quando a página estiver pronta
            document.addEventListener('DOMContentLoaded', function() {
                // Gráfico de modelos
                const modelsCtx = document.getElementById('modelsChart').getContext('2d');
                const modelsLabels = Object.keys(statsData.models);
                const modelsValues = Object.values(statsData.models);
                
                new Chart(modelsCtx, {
                    type: 'pie',
                    data: {
                        labels: modelsLabels,
                        datasets: [{
                            data: modelsValues,
                            backgroundColor: [
                                '#0047AB', '#1E90FF', '#87CEEB', '#4169E1', '#00BFFF', 
                                '#1E3A8A', '#3B82F6', '#7DD3FC', '#0EA5E9', '#0369A1'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'right',
                            },
                            title: {
                                display: false
                            }
                        }
                    }
                });
                
                // Gráfico de tags
                const tagsCtx = document.getElementById('tagsChart').getContext('2d');
                const tagsLabels = Object.keys(statsData.top_tags);
                const tagsValues = Object.values(statsData.top_tags);
                
                new Chart(tagsCtx, {
                    type: 'bar',
                    data: {
                        labels: tagsLabels,
                        datasets: [{
                            label: 'Número de Ocorrências',
                            data: tagsValues,
                            backgroundColor: '#0047AB',
                            borderColor: '#00008B',
                                    """Gera uma visualização HTML da memória."""
        memoria_id = template["metadata"]["id"]
        title = template["metadata"].get("title", "Memória de Chat")
        html_filename = f"{memoria_id}.html"
        html_path = os.path.join(MEMORIES_DIR, html_filename)
        
        # Obter dados para o template HTML
        metadata = template["metadata"]
        summary = template["summary"]
        metrics = template.get("metrics", {})
        messages = template["conversation"].get("messages", [])
        entities = template["semantic_structure"].get("entities", [])
        clusters = template["semantic_structure"].get("topic_clusters", [])
        
        # Criar HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="{metadata.get('language', 'pt-br')}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>JESUS CHAT MEMÓRIAS - {title}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet">
            <style>
                :root {
                    --primary: #0047AB;          /* Azul cobalto */
                    --primary-dark: #00008B;     /* Azul escuro */
                    --secondary: #1E90FF;        /* Azul Dodger */
                    --accent: #87CEEB;           /* Azul céu */
                    --light-blue: #E6F2FF;       /* Azul claro */
                    --dark: #000000;             /* Preto */
                    --dark-gray: #222222;        /* Cinza escuro */
                    --light: #FFFFFF;            /* Branco */
                    --gray: #F0F0F0;             /* Cinza claro */
                }
                
                body {{
                    font-family: 'Montserrat', sans-serif;
                    background-color: var(--gray);
                    color: var(--dark);
                }}
                
                .navbar {{
                    background-color: var(--dark);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                }}
                
                .navbar-brand {{
                    font-weight: 800;
                    font-size: 1.4rem;
                    color: var(--light) !important;
                }}
                
                .navbar-brand .highlight {{
                    color: var(--secondary);
                }}
                
                .memory-header {{
                    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                    color: var(--light);
                    padding: 3rem 0;
                    position: relative;
                }}
                
                .memory-metadata {{
                    background-color: var(--light);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-top: -2rem;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    margin-bottom: 2rem;
                }}
                
                .metadata-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 0.5rem;
                }}
                
                .metadata-item i {{
                    color: var(--primary);
                    margin-right: 0.5rem;
                    width: 20px;
                    text-align: center;
                }}
                
                .memory-content {{
                    background-color: var(--light);
                    border-radius: 10px;
                    padding: 2rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                    margin-bottom: 2rem;
                }}
                
                .memory-title {{
                    font-weight: 800;
                    color: var(--light);
                    margin-bottom: 1rem;
                }}
                
                .model-badge {{
                    background-color: rgba(255, 255, 255, 0.2);
                    color: var(--light);
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    display: inline-block;
                    margin-bottom: 1rem;
                }}
                
                .section-title {{
                    font-weight: 700;
                    margin-bottom: 1.5rem;
                    color: var(--primary-dark);
                    border-bottom: 2px solid var(--secondary);
                    padding-bottom: 0.5rem;
                }}
                
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 1rem;
                    margin-bottom: 2rem;
                }}
                
                .metric-card {{
                    flex: 1;
                    min-width: 150px;
                    background-color: var(--light-blue);
                    border-radius: 10px;
                    padding: 1rem;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .metric-value {{
                    font-size: 2rem;
                    font-weight: 800;
                    color: var(--primary);
                }}
                
                .metric-label {{
                    font-size: 0.9rem;
                    color: var(--dark-gray);
                    font-weight: 600;
                }}
                
                .tag-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    margin-bottom: 2rem;
                }}
                
                .tag {{
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}
                
                .message {{
                    margin-bottom: 1.5rem;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                
                .message-human {{
                    background-color: var(--light-blue);
                    border-left: 5px solid var(--primary);
                }}
                
                .message-assistant {{
                    background-color: white;
                    border-left: 5px solid var(--secondary);
                }}
                
                .message-header {{
                    padding: 0.8rem 1.5rem;
                    display: flex;
                    justify-content: space-between;
                    background-color: rgba(0, 0, 0, 0.05);
                }}
                
                .message-role {{
                    font-weight: 700;
                }}
                
                .message-time {{
                    font-size: 0.8rem;
                    color: var(--dark-gray);
                }}
                
                .message-content {{
                    padding: 1.5rem;
                    white-space: pre-wrap;
                }}
                
                .message-content pre {{
                    background-color: #f8f9fa;
                    padding: 1rem;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                
                .cluster-card {{
                    background-color: white;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .cluster-name {{
                    font-weight: 700;
                    margin-bottom: 1rem;
                    color: var(--primary-dark);
                }}
                
                .keyword {{
                    display: inline-block;
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.2rem 0.6rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    margin-right: 0.5rem;
                    margin-bottom: 0.5rem;
                }}
                
                .entity-card {{
                    display: flex;
                    align-items: center;
                    background-color: white;
                    border-radius: 10px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                
                .entity-icon {{
                    width: 40px;
                    height: 40px;
                    background-color: var(--primary);
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 1rem;
                }}
                
                .entity-details {{
                    flex: 1;
                }}
                
                .entity-name {{
                    font-weight: 700;
                    margin-bottom: 0.3rem;
                }}
                
                .entity-type {{
                    font-size: 0.8rem;
                    color: var(--dark-gray);
                }}
                
                .entity-mentions {{
                    background-color: var(--light-blue);
                    color: var(--primary-dark);
                    padding: 0.2rem 0.6rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                }}
                
                .summary-box {{
                    background-color: var(--light-blue);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 2rem;
                    border-left: 5px solid var(--primary);
                }}
                
                .insight-item {{
                    margin-bottom: 1rem;
                    padding-left: 1.5rem;
                    position: relative;
                }}
                
                .insight-item::before {{
                    content: "•";
                    color: var(--primary);
                    font-weight: bold;
                    position: absolute;
                    left: 0;
                }}
                
                .action-item {{
                    margin-bottom: 1rem;
                    padding-left: 1.5rem;
                    position: relative;
                }}
                
                .action-item::before {{
                    content: "✓";
                    color: var(--primary);
                    font-weight: bold;
                    position: absolute;
                    left: 0;
                }}
                
                .cross {{
                    width: 40px;
                    height: 40px;
                    margin-bottom: 1rem;
                    position: relative;
                }}
                
                .cross span {{
                    position: absolute;
                    background-color: var(--secondary);
                    border-radius: 2px;
                }}
                
                .cross span:nth-child(1) {{
                    width: 4px;
                    height: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                }}
                
                .cross span:nth-child(2) {{
                    width: 100%;
                    height: 4px;
                    top: 50%;
                    transform: translateY(-50%);
                }}
                
                .footer {{
                    background-color: var(--dark);
                    color: var(--light);
                    padding: 2rem 0;
                    text-align: center;
                }}
                
                .code {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 1rem;
                    overflow-x: auto;
                    font-family: monospace;
                    margin-bottom: 1rem;
                }}
            </style>
        </head>
        <body>
            <!-- Navbar -->
            <nav class="navbar navbar-dark">
                <div class="container">
                    <a class="navbar-brand" href="../index.html">
                        <span class="highlight">JESUS</span> CHAT MEMÓRIAS
                    </a>
                    <div>
                        <a href="../index.html" class="btn btn-outline-light btn-sm">
                            <i class="fas fa-home me-1"></i> Home
                        </a>
                    </div>
                </div>
            </nav>
            
            <!-- Memory Header -->
            <header class="memory-header">
                <div class="container">
                    <div class="model-badge">
                        <i class="fas fa-robot me-1"></i> {metadata.get('model', 'IA')}
                    </div>
                    <h1 class="memory-title">{title}</h1>
                    <p class="lead text-white opacity-75">
                        <i class="far fa-calendar-alt me-1"></i> {self._format_date(metadata.get('timestamp', ''))}
                    </p>
                </div>
            </header>
            
            <!-- Main Content -->
            <div class="container py-4">
                <!-- Metadata -->
                <div class="memory-metadata">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-tag"></i>
                                <span>ID: {memoria_id}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-language"></i>
                                <span>Idioma: {metadata.get('language', 'Não especificado')}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-comment-dots"></i>
                                <span>Mensagens: {len(messages)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-clock"></i>
                                <span>Duração: {metadata.get('duration_seconds', 0)} segundos</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-calculator"></i>
                                <span>Tokens: {metadata.get('total_tokens', 0)}</span>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="metadata-item">
                                <i class="fas fa-server"></i>
                                <span>Plataforma: {metadata.get('platform', 'Não especificada')}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Summary -->
                <div class="memory-content">
                    <h2 class="section-title">Resumo</h2>
                    <div class="summary-box">
                        <p class="mb-0">{summary.get('brief', 'Não há resumo disponível.')}</p>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h3 class="section-title h5">Insights Principais</h3>
                            <div class="insights-container">
                                {self._generate_insights_html(summary.get('key_insights', []))}
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h3 class="section-title h5">Ações Recomendadas</h3>
                            <div class="actions-container">
                                {self._generate_actions_html(summary.get('action_items', []))}
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Metrics -->
                <div class="memory-content">
                    <h2 class="section-title">Métricas</h2>
                    <div class="metrics-container">
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('problem_resolution_score', 0))}</div>
                            <div class="metric-label">Resolução</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('technical_accuracy', 0))}</div>
                            <div class="metric-label">Precisão</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('response_completeness', 0))}</div>
                            <div class="metric-label">Completude</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._format_percentage(metrics.get('chat_efficiency', 0))}</div>
                            <div class="metric-label">Eficiência</div>
                        </div>
                    </div>
                </div>
                
                <!-- Tags -->
                <div class="memory-content">
                    <h2 class="section-title">Tags</h2>
                    <div class="tag-container">
                        {self._generate_tags_html(metadata.get('tags', []))}
                    </div>
                </div>
                
                <!-- Clusters -->
                <div class="memory-content">
                    <h2 class="section-title">Clusters Temáticos</h2>
                    <div class="clusters-container">
                        {self._generate_clusters_html(clusters)}
                    </div>
                </div>
                
                <!-- Entities -->
                <div class="memory-content">
                    <h2 class="section-title">Entidades</h2>
                    <div class="entities-container">
                        {self._generate_entities_html(entities)}
                    </div>
                </div>
                
                <!-- Chat -->
                <div class="memory-content">
                    <h2 class="section-title">Conversa</h2>
                    <div class="conversation-container">
                        {self._generate_messages_html(messages)}
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <footer class="footer">
                <div class="container">
                    <div class="cross">
                        <span></span>
                        <span></span>
                    </div>
                    <p class="mb-0">JESUS CHAT MEMÓRIAS &copy; {datetime.datetime.now().year}</p>
                    <small>Memória gerada em {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}</small>
                </div>
            </footer>
        </body>
        </html>
        """
        
        # Salvar arquivo HTML
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _format_date(self, timestamp: str) -> str:
        """Formata timestamp para exibição."""
        if not timestamp:
            return "Data não especificada"
        
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%d/%m/%Y %H:%M')
        except (ValueError, TypeError):
            return timestamp
    
    def _format_percentage(self, value: float) -> str:
        """Formata valor de métrica como porcentagem."""
        if not value:
            return "0%"
        
        return f"{int(value * 100)}%"
    
    def _generate_insights_html(self, insights: List[str]) -> str:
        """Gera HTML para insights."""
        if not insights:
            return "<p>Nenhum insight disponível.</p>"
        
        html = ""
        for insight in insights:
            html += f'<div class="insight-item">{insight}</div>'
        
        return html
    
    def _generate_actions_html(self, actions: List[str]) -> str:
        """Gera HTML para ações recomendadas."""
        if not actions:
            return "<p>Nenhuma ação recomendada.</p>"
        
        html = ""
        for action in actions:
            html += f'<div class="action-item">{action}</div>'
        
        return html
    
    def _generate_tags_html(self, tags: List[str]) -> str:
        """Gera HTML para tags."""
        if not tags:
            return "<span class='tag'>sem-tags</span>"
        
        html = ""
        for tag in tags:
            html += f'<span class="tag">{tag}</span>'
        
        return html
    
    def _generate_clusters_html(self, clusters: List[Dict]) -> str:
        """Gera HTML para clusters temáticos."""
        if not clusters:
            return "<p>Nenhum cluster temático identificado.</p>"
        
        html = ""
        for cluster in clusters:
            keywords_html = ""
            for keyword in cluster.get("keywords", []):
                keywords_html += f'<span class="keyword">{keyword}</span>'
            
            importance = cluster.get("importance", 0)
            importance_percent = self._format_percentage(importance)
            
            html += f"""
            <div class="cluster-card">
                <div class="d-flex justify-content-between align-items-center">
                    <h3 class="cluster-name">{cluster.get("name", "Sem nome")}</h3>
                    <span class="entity-mentions">Relevância: {importance_percent}</span>
                </div>
                <div class="keywords-container">
                    {keywords_html}
                </div>
            </div>
            """
        
        return html
    
    def _generate_entities_html(self, entities: List[Dict]) -> str:
        """Gera HTML para entidades."""
        if not entities:
            return "<p>Nenhuma entidade identificada.</p>"
        
        html = ""
        for entity in entities:
            entity_type = entity.get("type", "conceito")
            icon = self._get_entity_icon(entity_type)
            
            html += f"""
            <div class="entity-card">
                <div class="entity-icon">
                    <i class="{icon}"></i>
                </div>
                <div class="entity-details">
                    <div class="entity-name">{entity.get("name", "Sem nome")}</div>
                    <div class="entity-type">Tipo: {entity_type}</div>
                </div>
                <span class="entity-mentions">{entity.get("mentions", 0)} menções</span>
            </div>
            """
        
        return html
    
    def _get_entity_icon(self, entity_type: str) -> str:
        """Retorna ícone baseado no tipo de entidade."""
        icons = {
            "technology": "fas fa-microchip",
            "language": "fas fa-code",
            "concept": "fas fa-lightbulb",
            "person": "fas fa-user",
            "organization": "fas fa-building",
            "location": "fas fa-map-marker-alt",
            "protocol": "fas fa-shield-alt",
            "application": "fas fa-cube",
            "framework": "fas fa-layer-group",
            "architecture": "fas fa-sitemap"
        }
        
        return icons.get(entity_type.lower(), "fas fa-tag")
    
    def _generate_messages_html(self, messages: List[Dict]) -> str:
        """Gera HTML para mensagens."""
        if not messages:
            return "<p>Nenhuma mensagem disponível.</p>"
        
        html = ""
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = self._format_date(message.get("timestamp", ""))
            
            # Formatação de código em blocos de código
            content = self._format_code_blocks(content)
            
            message_class = "message-human" if role == "human" else "message-assistant"
            role_display = "Usuário" if role == "human" else "Assistente"
            
            html += f"""
            <div class="message {message_class}">
                <div class="message-header">
                    <div class="message-role">{role_display}</div>
                    <div class="message-time">{timestamp}</div>
                </div>
                <div class="message-content">
                    {content}
                </div>
            </div>
            """
        
        return html
    
    def _format_code_blocks(self, content: str) -> str:
        """Formata blocos de código no conteúdo."""
        # Padrão para corresponder a blocos de código markdown
        pattern = r'```(.*?)\n(.*?)```'
        
        def replace_code_block(match):
            language = match.group(1).strip()
            code = match.group(2)
            return f'<pre><code class="language-{language}">{code}</code></pre>'
        
        # Substituir blocos de código
        formatted_content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)
        
        return formatted_content
    
    def search_chats(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats com base em uma consulta textual.
        
        Args:
            query: Texto para busca
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        # Busca por título, resumo e tags
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        WHERE 
            m.title LIKE ? OR 
            m.summary_brief LIKE ? OR 
            m.tags LIKE ?
        ORDER BY m.timestamp DESC
        LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def search_by_entity(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats que mencionam uma entidade específica.
        
        Args:
            entity_name: Nome da entidade
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        JOIN entities e ON m.id = e.memoria_id
        WHERE e.name LIKE ?
        ORDER BY e.mentions DESC
        LIMIT ?
        """, (f"%{entity_name}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def search_by_cluster(self, cluster_topic: str, limit: int = 10) -> List[Dict]:
        """
        Busca chats relacionados a um cluster/tópico específico.
        
        Args:
            cluster_topic: Tópico ou palavra-chave do cluster
            limit: Número máximo de resultados
            
        Returns:
            Lista de chats correspondentes
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT m.id, m.title, m.model, m.timestamp, m.summary_brief, m.tags, m.filepath_html
        FROM memorias m
        JOIN clusters c ON m.id = c.memoria_id
        WHERE c.name LIKE ? OR c.keywords LIKE ?
        ORDER BY c.importance DESC
        LIMIT ?
        """, (f"%{cluster_topic}%", f"%{cluster_topic}%", limit))
        
        results = cursor.fetchall()
        
        # Formatar resultados
        chats = []
        for row in results:
            chat = {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "timestamp": row[3],
                "summary": row[4],
                "tags": row[5].split(",") if row[5] else [],
                "html_path": row[6]
            }
            chats.append(chat)
        
        return chats
    
    def export_to_csv(self, output_path: str = "jesus_chat_memorias_export.csv") -> str:
        """
        Exporta o banco de dados de memórias para um arquivo CSV.
        
        Args:
            output_path: Caminho para salvar o arquivo CSV
            
        Returns:
            Caminho do arquivo CSV exportado
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT 
            m.id, m.timestamp, m.title, m.model, m.language, m.import os
import json
import uuid
import hashlib
import datetime
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import numpy as np

# Configurações
DB_PATH = "db/jesus_chat_memorias.sqlite"
MEMORIES_DIR = "memorias"
TEMPLATES_DIR = "templates"
HTML_TEMPLATE_PATH = "app/templates/memory_template.html"

class MemoriaChatProcessor:
    """
    Processador para memórias de chat com IAs.
    Processa templates JSON e cria representações em HTML e banco de dados.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """Inicializa o processador."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(MEMORIES_DIR, exist_ok=True)
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        
        self.db_path = db_path
        self.conn = self._init_database()
    
    def _init_database(self) -> sqlite3.Connection:
        """Inicializa o banco de dados SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Criar tabela memorias
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memorias (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            title TEXT,
            model TEXT,
            language TEXT,
            summary_brief TEXT,
            problem_resolution_score REAL,
            response_completeness REAL,
            technical_accuracy REAL,
            chat_efficiency REAL,
            tags TEXT,
            filepath_csv TEXT,
            filepath_html TEXT,
            created_at TEXT
        )
        """)
        
        # Criar tabela clusters
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            name TEXT,
            keywords TEXT,
            importance REAL,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela entities
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            mentions INTEGER,
            related_clusters TEXT,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela knowledge_graph (nós)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            type TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela edges (arestas)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY,
            source TEXT,
            target TEXT,
            relationship TEXT,
            weight REAL,
            memoria_id TEXT,
            FOREIGN KEY (source) REFERENCES graph_nodes (id),
            FOREIGN KEY (target) REFERENCES graph_nodes (id),
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar tabela messages
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            memoria_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            tokens INTEGER,
            clusters TEXT,
            sentiment TEXT,
            intent TEXT,
            key_points TEXT,
            FOREIGN KEY (memoria_id) REFERENCES memorias (id)
        )
        """)
        
        # Criar índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id ON clusters (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_entities ON entities (memoria_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memoria_id_messages ON messages (memoria_id)")
        
        conn.commit()
        return conn
    
    def process_template(self, template_path: str) -> str:
        """
        Processa um template JSON de chat e armazena no banco de dados.
        
        Args:
            template_path: Caminho para o arquivo JSON do template
            
        Returns:
            ID da memória processada
        """
        # Carregar template
        with open(template_path, 'r', encoding='utf-8') as f:
            try:
                template = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Erro ao decodificar o template JSON: {e}")
        
        # Validar template
        self._validate_template(template)
        
        # Obter ID da memória
        memoria_id = template["metadata"]["id"]
        if not memoria_id:
            memoria_id = f"chat_{uuid.uuid4()}"
            template["metadata"]["id"] = memoria_id
        
        # Processar e salvar no banco de dados
        self._save_to_database(template)
        
        # Gerar arquivos
        csv_path = self._generate_csv_export(template)
        html_path = self._generate_html_view(template)
        
        # Atualizar caminhos de arquivo
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memorias SET filepath_csv = ?, filepath_html = ? WHERE id = ?",
            (csv_path, html_path, memoria_id)
        )
        self.conn.commit()
        
        print(f"✅ Memória processada e salva com sucesso: {memoria_id}")
        return memoria_id
    
    def _validate_template(self, template: Dict) -> bool:
        """Valida a estrutura do template."""
        required_sections = ["metadata", "semantic_structure", "conversation", "summary"]
        for section in required_sections:
            if section not in template:
                raise ValueError(f"Seção obrigatória ausente no template: {section}")
        
        # Validar metadata
        required_metadata = ["id", "title", "timestamp"]
        for field in required_metadata:
            if field not in template["metadata"]:
                template["metadata"][field] = "" if field != "timestamp" else datetime.datetime.now().isoformat()
        
        return True
    
    def _save_to_database(self, template: Dict) -> None:
        """Salva o template processado no banco de dados."""
        cursor = self.conn.cursor()
        memoria_id = template["metadata"]["id"]
        
        # Salvar memória principal
        tags = ",".join(template["metadata"].get("tags", []))
        metrics = template.get("metrics", {})
        
        cursor.execute("""
        INSERT OR REPLACE INTO memorias (
            id, timestamp, title, model, language, summary_brief, 
            problem_resolution_score, response_completeness, technical_accuracy, 
            chat_efficiency, tags, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memoria_id,
            template["metadata"].get("timestamp", ""),
            template["metadata"].get("title", ""),
            template["metadata"].get("model", ""),
            template["metadata"].get("language", ""),
            template["summary"].get("brief", ""),
            metrics.get("problem_resolution_score", 0),
            metrics.get("response_completeness", 0),
            metrics.get("technical_accuracy", 0),
            metrics.get("chat_efficiency", 0),
            tags,
            datetime.datetime.now().isoformat()
        ))
        
        # Salvar clusters
        for cluster in template["semantic_structure"].get("topic_clusters", []):
            cluster_id = cluster.get("id", f"cluster_{uuid.uuid4()}")
            keywords = ",".join(cluster.get("keywords", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO clusters (id, memoria_id, name, keywords, importance)
            VALUES (?, ?, ?, ?, ?)
            """, (
                cluster_id,
                memoria_id,
                cluster.get("name", ""),
                keywords,
                cluster.get("importance", 0)
            ))
        
        # Salvar entidades
        for entity in template["semantic_structure"].get("entities", []):
            entity_id = f"entity_{hashlib.md5(entity.get('name', '').encode()).hexdigest()[:8]}"
            related_clusters = ",".join(entity.get("related_clusters", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO entities (id, name, type, mentions, related_clusters, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entity_id,
                entity.get("name", ""),
                entity.get("type", ""),
                entity.get("mentions", 0),
                related_clusters,
                memoria_id
            ))
        
        # Salvar nós do grafo
        for node in template["semantic_structure"].get("knowledge_graph", {}).get("nodes", []):
            cursor.execute("""
            INSERT OR REPLACE INTO graph_nodes (id, label, type, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?)
            """, (
                node.get("id", ""),
                node.get("label", ""),
                node.get("type", ""),
                node.get("weight", 0),
                memoria_id
            ))
        
        # Salvar arestas do grafo
        for idx, edge in enumerate(template["semantic_structure"].get("knowledge_graph", {}).get("edges", [])):
            edge_id = f"edge_{idx}_{memoria_id}"
            cursor.execute("""
            INSERT OR REPLACE INTO graph_edges (id, source, target, relationship, weight, memoria_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                edge.get("source", ""),
                edge.get("target", ""),
                edge.get("relationship", ""),
                edge.get("weight", 0),
                memoria_id
            ))
        
        # Salvar mensagens
        for message in template["conversation"].get("messages", []):
            message_id = message.get("id", f"msg_{uuid.uuid4()}")
            clusters = ",".join(message.get("clusters", []))
            key_points = ",".join(message.get("key_points", []))
            
            cursor.execute("""
            INSERT OR REPLACE INTO messages (
                id, memoria_id, role, content, timestamp, tokens, 
                clusters, sentiment, intent, key_points
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                memoria_id,
                message.get("role", ""),
                message.get("content", ""),
                message.get("timestamp", ""),
                message.get("tokens", 0),
                clusters,
                message.get("sentiment", ""),
                message.get("intent", ""),
                key_points
            ))
        
        self.conn.commit()
    
    def _generate_csv_export(self, template: Dict) -> str:
        """Gera um arquivo CSV da memória."""
        memoria_id = template["metadata"]["id"]
        csv_filename = f"{memoria_id}.csv"
        csv_path = os.path.join(MEMORIES_DIR, csv_filename)
        
        # Criar DataFrame para mensagens
        messages = []
        for msg in template["conversation"].get("messages", []):
            messages.append({
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
                "tokens": msg.get("tokens", 0),
                "sentiment": msg.get("sentiment", ""),
                "intent": msg.get("intent", ""),
                "key_points": ", ".join(msg.get("key_points", []))
            })
        
        df = pd.DataFrame(messages)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return csv_path
    
    def _generate_html_view(self, template: Dict) -> str:
        """Gera uma visualização HTML da memória."""
        memoria_id = template["metadata"]["id"]
        title = template["metadata"].get
