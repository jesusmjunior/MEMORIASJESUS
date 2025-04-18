import os
import json
import sqlite3
from datetime import datetime

class MemoriaChatProcessor:
    def __init__(self, db_path="db/jesus_chat_memorias.sqlite"):
        self.db_path = db_path
        os.makedirs("memorias", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        os.makedirs("db", exist_ok=True)

    def process_template(self, chat_data):
        summary = chat_data.get("summary", {})
        metadata = self.extract_metadata(chat_data)
        html_view = self.build_html_view(chat_data)

        self.save_to_disk(html_view, metadata)
        self.save_to_db(chat_data, summary, metadata)

        return {
            "summary": summary,
            "metadata": metadata,
            "html_view": html_view
        }

    def extract_metadata(self, chat_data):
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages_count": len(chat_data.get("conversation", {}).get("messages", [])),
            "model": chat_data.get("chat_metadata", {}).get("model", "Desconhecido"),
            "title": chat_data.get("chat_metadata", {}).get("title", "Sem Título"),
            "tags": ','.join(chat_data.get("chat_metadata", {}).get("tags", []))
        }

    def build_html_view(self, chat_data):
        summary = chat_data.get("summary", {})
        metadata = self.extract_metadata(chat_data)
        html = f"""
        <html>
            <head><title>Memória</title></head>
            <body>
                <h1>{metadata['title']}</h1>
                <p><b>Modelo:</b> {metadata['model']}</p>
                <p><b>Tags:</b> {metadata['tags']}</p>
                <p><b>Resumo:</b> {summary.get('brief', '')}</p>
                <h2>Entidades</h2>
                <ul>
                    {''.join([f'<li>{e.get('name')} ({e.get('type')})</li>' for e in summary.get('entities', [])])}
                </ul>
                <h2>Clusters</h2>
                <ul>
                    {''.join([f'<li>{c.get('name')}: {c.get('keywords')}</li>' for c in summary.get('topic_clusters', [])])}
                </ul>
                <h2>Métricas</h2>
                <ul>
                    {''.join([f'<li>{k}: {v}</li>' for k, v in summary.get('metrics', {}).items()])}
                </ul>
                <h2>Artefatos</h2>
                <ul>
                    {''.join([f'<li>{a.get('name')}: {a.get('type')}</li>' for a in chat_data.get('conversation', {}).get('artifacts', [])])}
                </ul>
                <p><b>Data:</b> {metadata['timestamp']}</p>
            </body>
        </html>
        """
        return html

    def save_to_disk(self, html_view, metadata):
        filename = f"memorias/memoria_{metadata['timestamp'].replace(':', '-')}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_view)

    def save_to_db(self, chat_data, summary, metadata):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Tabela principal
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memoria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titulo TEXT,
                modelo TEXT,
                tags TEXT,
                resumo TEXT,
                data TEXT
            )
        """)
        cur.execute("""
            INSERT INTO memoria (titulo, modelo, tags, resumo, data)
            VALUES (?, ?, ?, ?, ?)
        """, (metadata["title"], metadata["model"], metadata["tags"], summary.get("brief", ""), metadata["timestamp"]))
        memoria_id = cur.lastrowid

        # Entidades
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memoria_id INTEGER,
                name TEXT,
                type TEXT,
                mentions INTEGER
            )
        """)
        for ent in summary.get("entities", []):
            cur.execute("INSERT INTO entities (memoria_id, name, type, mentions) VALUES (?, ?, ?, ?)",
                        (memoria_id, ent.get("name"), ent.get("type"), ent.get("mentions", 1)))

        # Clusters
        cur.execute("""
            CREATE TABLE IF NOT EXISTS topic_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memoria_id INTEGER,
                name TEXT,
                keywords TEXT,
                importance REAL
            )
        """)
        for cluster in summary.get("topic_clusters", []):
            cur.execute("INSERT INTO topic_clusters (memoria_id, name, keywords, importance) VALUES (?, ?, ?, ?)",
                        (memoria_id, cluster.get("name"), cluster.get("keywords"), cluster.get("importance", 0.0)))

        conn.commit()
        conn.close()
