import os
import json
import uuid
import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from werkzeug.utils import secure_filename
# Importar o processador de memórias
from memory_processor import MemoriaChatProcessor, process_directory,
generate_index_html
# Configuração
UPLOAD_FOLDER = 'templates'
ALLOWED_EXTENSIONS = {'json'}
STATIC_FOLDER = 'memorias'
app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload
# Inicializar processador
processor = MemoriaChatProcessor()
def allowed_file(filename):
"""Verifica se a extensão do arquivo é permitida."""
return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def index():
"""Rota para a página inicial."""
index_path = 'index.html'
if not os.path.exists(index_path):
# Gerar página inicial se não existir
generate_index_html(processor, index_path)
return send_from_directory('.', index_path)
@app.route('/<path:path>')
def serve_file(path):
"""Serve arquivos estáticos."""
if os.path.exists(path):
return send_from_directory('.', path)
else:
return "Arquivo não encontrado", 404
@app.route('/memorias/<path:path>')
def serve_memory(path):
"""Serve arquivos de memória."""
return send_from_directory(STATIC_FOLDER, path)
@app.route('/api/upload', methods=['POST'])
def upload_file():
"""Endpoint para upload de arquivo JSON."""
if 'file' not in request.files:
return jsonify({'success': False, 'error': 'Nenhum arquivo enviado'}), 400
file = request.files['file']
if file.filename == '':
return jsonify({'success': False, 'error': 'Nome de arquivo vazio'}), 400
if not allowed_file(file.filename):
return jsonify({'success': False, 'error': 'Tipo de arquivo não permitido. Use .json'}), 400
try:
# Salvar arquivo
filename = secure_filename(file.filename)
filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
file.save(filepath)
# Processar template
memoria_id = processor.process_template(filepath)
# Regenerar página inicial
generate_index_html(processor)
return jsonify({
'success': True,
'message': 'Template processado com sucesso',
'memoria_id': memoria_id
})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/memory/<memory_id>')
def get_memory(memory_id):
"""Endpoint para obter detalhes de uma memória específica."""
try:
memory = processor.get_chat_by_id(memory_id)
if memory:
return jsonify({'success': True, 'data': memory})
else:
return jsonify({'success': False, 'error': 'Memória não encontrada'}), 404
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/search')
def search_memories():
"""Endpoint para busca de memórias."""
query = request.args.get('q', '')
limit = int(request.args.get('limit', 10))
try:
results = processor.search_chats(query, limit)
return jsonify({'success': True, 'data': results})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/entity/<entity_name>')
def search_by_entity(entity_name):
"""Endpoint para busca por entidade."""
limit = int(request.args.get('limit', 10))
try:
results = processor.search_by_entity(entity_name, limit)
return jsonify({'success': True, 'data': results})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/cluster/<cluster_topic>')
def search_by_cluster(cluster_topic):
"""Endpoint para busca por cluster/tópico."""
limit = int(request.args.get('limit', 10))
try:
results = processor.search_by_cluster(cluster_topic, limit)
return jsonify({'success': True, 'data': results})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/stats')
def get_stats():
"""Endpoint para obter estatísticas do sistema."""
try:
stats = processor.get_stats()
return jsonify({'success': True, 'data': stats})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/graph')
def get_graph():
"""Endpoint para obter dados do grafo de conhecimento."""
try:
graph_data = processor.generate_graph_data()
return jsonify({'success': True, 'data': graph_data})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/wordcloud')
def get_wordcloud():
"""Endpoint para obter dados de nuvem de palavras."""
try:
wordcloud_data = processor.generate_word_cloud_data()
return jsonify({'success': True, 'data': wordcloud_data})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/process_json', methods=['POST'])
def process_json_data():
"""Endpoint para processar dados JSON enviados diretamente."""
try:
if not request.is_json:
return jsonify({'success': False, 'error': 'Dados não estão em formato JSON'}), 400
# Obter dados JSON
template = request.get_json()
# Salvar em arquivo temporário
temp_filename = f"temp_{uuid.uuid4()}.json"
temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
with open(temp_filepath, 'w', encoding='utf-8') as f:
json.dump(template, f, ensure_ascii=False, indent=2)
# Processar template
memoria_id = processor.process_template(temp_filepath)
# Regenerar página inicial
generate_index_html(processor)
return jsonify({
'success': True,
'message': 'Template processado com sucesso',
'memoria_id': memoria_id
})
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/export_csv')
def export_csv():
"""Endpoint para exportar banco de dados para CSV."""
try:
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"jesus_chat_memorias_export_{timestamp}.csv"
csv_path = processor.export_to_csv(output_path)
# Enviar arquivo para download
return send_from_directory('.', output_path, as_attachment=True)
except Exception as e:
return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/upload')
def upload_page():
"""Página para upload de templates JSON."""
html = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Upload - JESUS CHAT MEMÓRIAS</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
rel="stylesheet">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link
href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display
=swap" rel="stylesheet">
<style>
:root {
--primary: #0047AB;
--primary-dark: #00008B;
--secondary: #1E90FF;
--accent: #87CEEB;
--light-blue: #E6F2FF;
--dark: #000000;
--dark-gray: #222222;
--light: #FFFFFF;
--gray: #F0F0F0;
}
body {
font-family: 'Montserrat', sans-serif;
background-color: var(--gray);
color: var(--dark);
padding-top: 56px;
}
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
.header-section {
background: linear-gradient(135deg, var(--dark) 0%, var(--primary-dark) 100%);
color: var(--light);
padding: 5rem 0 3rem;
}
.upload-container {
background: var(--light);
border-radius: 15px;
padding: 3rem;
box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
margin-top: -2rem;
position: relative;
z-index: 10;
}
.upload-area {
border: 2px dashed var(--primary);
border-radius: 10px;
padding: 3rem;
text-align: center;
margin-bottom: 2rem;
transition: all 0.3s;
background-color: var(--light-blue);
}
.upload-area.highlight {
background-color: rgba(0, 71, 171, 0.1);
}
.upload-icon {
font-size: 3rem;
color: var(--primary);
margin-bottom: 1rem;
}
.btn-primary {
background-color: var(--primary);
border-color: var(--primary);
}
.btn-primary:hover {
background-color: var(--primary-dark);
border-color: var(--primary-dark);
}
.footer {
background-color: var(--dark);
color: var(--light);
padding: 2rem 0;
text-align: center;
margin-top: 4rem;
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
}
.cross span:nth-child(2) {
width: 100%;
height: 4px;
top: 50%;
transform: translateY(-50%);
}
.json-preview {
max-height: 400px;
overflow-y: auto;
background-color: #f8f9fa;
border-radius: 5px;
padding: 1rem;
margin-bottom: 1rem;
display: none;
}
.alert {
display: none;
margin-bottom: 1rem;
}
</style>
</head>
<body>
<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark fixed-top">
<div class="container">
<a class="navbar-brand" href="/">
<span class="highlight">JESUS</span> CHAT MEMÓRIAS
</a>
<button class="navbar-toggler" type="button" data-bs-toggle="collapse"
data-bs-target="#navbarNav">
<span class="navbar-toggler-icon"></span>
</button>
<div class="collapse navbar-collapse" id="navbarNav">
<ul class="navbar-nav ms-auto">
<li class="nav-item">
<a class="nav-link" href="/">
<i class="fas fa-home me-1"></i> Home
</a>
</li>
<li class="nav-item">
<a class="nav-link" href="/search.html">
<i class="fas fa-search me-1"></i> Buscar
</a>
</li>
<li class="nav-item">
<a class="nav-link" href="/entities.html">
<i class="fas fa-tags me-1"></i> Entidades
</a>
</li>
<li class="nav-item">
<a class="nav-link" href="/graph.html">
<i class="fas fa-project-diagram me-1"></i> Grafo
</a>
</li>
<li class="nav-item">
<a class="nav-link active" href="/upload">
<i class="fas fa-upload me-1"></i> Upload
</a>
</li>
</ul>
</div>
</div>
</nav>
<!-- Header -->
<section class="header-section">
<div class="container">
<h1 class="text-center mb-4">Upload de Memórias</h1>
<p class="text-center lead">Adicione novas conversas ao sistema Jesus Chat
Memórias</p>
</div>
</section>
<!-- Upload Section -->
<div class="container">
<div class="upload-container">
<div class="alert alert-success" id="successAlert">
<i class="fas fa-check-circle me-2"></i> <span id="successMessage"></span>
</div>
<div class="alert alert-danger" id="errorAlert">
<i class="fas fa-exclamation-triangle me-2"></i> <span
id="errorMessage"></span>
</div>
<div class="mb-4">
<h2 class="h4 mb-3">Enviar Arquivo JSON</h2>
<div class="upload-area" id="dropArea">
<i class="fas fa-cloud-upload-alt upload-icon"></i>
<h3 class="h5 mb-3">Arraste o arquivo JSON ou clique para selecionar</h3>
<p class="text-muted">Formato aceito: .json (Max. 16MB)</p>
<input type="file" id="fileInput" accept=".json" class="d-none">
<button type="button" class="btn btn-primary mt-3" id="selectFileBtn">
<i class="fas fa-file-upload me-2"></i>Selecionar Arquivo
</button>
</div>
</div>
<div id="previewSection" style="display: none;">
<h3 class="h4 mb-3">Pré-visualização</h3>
<div class="json-preview" id="jsonPreview"></div>
<button type="button" class="btn btn-primary" id="processBtn">
<i class="fas fa-cogs me-2"></i>Processar Arquivo
</button>
<button type="button" class="btn btn-secondary ms-2" id="cancelBtn">
<i class="fas fa-times me-2"></i>Cancelar
</button>
</div>
<div class="mt-5">
<h3 class="h4 mb-3">Alternativa: Cole o JSON Diretamente</h3>
<div class="mb-3">
<textarea class="form-control" id="jsonText" rows="10" placeholder="Cole o
conteúdo JSON aqui..."></textarea>
</div>
<button type="button" class="btn btn-primary" id="processJsonBtn">
<i class="fas fa-cogs me-2"></i>Processar JSON
</button>
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
<p class="mb-0">JESUS CHAT MEMÓRIAS &copy; 2025</p>
</div>
</footer>
<script
src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
// Elementos do DOM
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const jsonPreview = document.getElementById('jsonPreview');
const previewSection = document.getElementById('previewSection');
const processBtn = document.getElementById('processBtn');
const cancelBtn = document.getElementById('cancelBtn');
const jsonText = document.getElementById('jsonText');
const processJsonBtn = document.getElementById('processJsonBtn');
const successAlert = document.getElementById('successAlert');
const errorAlert = document.getElementById('errorAlert');
const successMessage = document.getElementById('successMessage');
const errorMessage = document.getElementById('errorMessage');
// Variáveis globais
let selectedFile = null;
// Event listeners para drag and drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
dropArea.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) {
e.preventDefault();
e.stopPropagation();
}
['dragenter', 'dragover'].forEach(eventName => {
dropArea.addEventListener(eventName, highlight, false);
});
['dragleave', 'drop'].forEach(eventName => {
dropArea.addEventListener(eventName, unhighlight, false);
});
function highlight() {
dropArea.classList.add('highlight');
}
function unhighlight() {
dropArea.classList.remove('highlight');
}
// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);
function handleDrop(e) {
const dt = e.dataTransfer;
const files = dt.files;
if (files.length === 1) {
handleFiles(files);
}
}
// Handle selected files
selectFileBtn.addEventListener('click', () => {
fileInput.click();
});
fileInput.addEventListener('change', (e) => {
handleFiles(e.target.files);
});
function handleFiles(files) {
if (files.length === 0) return;
const file = files[0];
// Check if file is JSON
if (!file.type.match('application/json') && !file.name.endsWith('.json')) {
showError('Por favor, selecione um arquivo JSON válido.');
return;
}
selectedFile = file;
// Read file
const reader = new FileReader();
reader.onload = function(e) {
try {
const json = JSON.parse(e.target.result);
const jsonString = JSON.stringify(json, null, 2);
jsonPreview.textContent = jsonString;
previewSection.style.display = 'block';
jsonPreview.style.display = 'block';
hideAlerts();
} catch (error) {
showError('JSON inválido! Verifique o formato do arquivo.');
}
};
reader.readAsText(file);
}
// Process file
processBtn.addEventListener('click', () => {
if (!selectedFile) {
showError('Nenhum arquivo selecionado!');
return;
}
const formData = new FormData();
formData.append('file', selectedFile);
fetch('/api/upload', {
method: 'POST',
body: formData
})
.then(response => response.json())
.then(data => {
if (data.success) {
showSuccess(`Arquivo processado com sucesso! ID: ${data.memoria_id}`);
resetForm();
} else {
showError(data.error || 'Erro ao processar arquivo.');
}
})
.catch(error => {
showError('Erro ao enviar arquivo: ' + error.message);
});
});
// Process JSON text
processJsonBtn.addEventListener('click', () => {
const text = jsonText.value.trim();
if (!text) {
showError('Por favor, insira o conteúdo JSON.');
return;
}
try {
const json = JSON.parse(text);
fetch('/api/process_json', {
method: 'POST',
headers: {
'Content-Type': 'application/json'
},
body: JSON.stringify(json)
})
.then(response => response.json())
.then(data => {
if (data.success) {
showSuccess(`JSON processado com sucesso! ID: ${data.memoria_id}`);
jsonText.value = '';
} else {
showError(data.error || 'Erro ao processar JSON.');
}
})
.catch(error => {
showError('Erro ao enviar JSON: ' + error.message);
});
} catch (error) {
showError('JSON inválido! Verifique o formato.');
}
});
// Cancel selection
cancelBtn.addEventListener('click', resetForm);
function resetForm() {
selectedFile = null;
fileInput.value = '';
previewSection.style.display = 'none';
jsonPreview.textContent = '';
}
function showSuccess(message) {
successMessage.textContent = message;
successAlert.style.display = 'block';
errorAlert.style.display = 'none';
// Scroll to alert
successAlert.scrollIntoView({ behavior: 'smooth' });
// Hide after 5 seconds
setTimeout(() => {
successAlert.style.display = 'none';
}, 5000);
}
function showError(message) {
errorMessage.textContent = message;
errorAlert.style.display = 'block';
successAlert.style.display = 'none';
// Scroll to alert
errorAlert.scrollIntoView({ behavior: 'smooth' });
}
function hideAlerts() {
successAlert.style.display = 'none';
errorAlert.style.display = 'none';
}
</script>
</body>
</html>
"""
return render_template_string(html)
@app.errorhandler(404)
def page_not_found(e):
"""Página 404 personalizada."""
return render_template_string("""
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Página Não Encontrada - JESUS CHAT MEMÓRIAS</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
rel="stylesheet">
<style>
body {
font-family: 'Montserrat', sans-serif;
background-color: #f5f5f5;
color: #333;
display: flex;
align-items: center;
justify-content: center;
min-height: 100vh;
padding: 20px;
}
.error-container {
text-align: center;
max-width: 600px;
background-color: white;
border-radius: 10px;
padding: 40px;
box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}
.error-icon {
font-size: 5rem;
color: #0047AB;
margin-bottom: 2rem;
}
.error-title {
font-size: 2rem;
font-weight: 700;
margin-bottom: 1rem;
}
.error-message {
font-size: 1.2rem;
margin-bottom: 2rem;
color: #666;
}
.btn-primary {
background-color: #0047AB;
border-color: #0047AB;
}
.btn-primary:hover {
background-color: #00008B;
border-color: #00008B;
}
</style>
</head>
<body>
<div class="error-container">
<div class="error-icon">
<i class="fas fa-exclamation-triangle"></i>
</div>
<h1 class="error-title">Página Não Encontrada</h1>
<p class="error-message">A página que você está procurando não existe ou foi
movida.</p>
<a href="/" class="btn btn-primary">
<i class="fas fa-home me-2"></i>Voltar para a Página Inicial
</a>
</div>
</body>
</html>
"""), 404
@app.errorhandler(500)
def server_error(e):
"""Página 500 personalizada."""
return render_template_string("""
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Erro
