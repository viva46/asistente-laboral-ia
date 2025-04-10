import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import os
import pickle
import tempfile
from sentence_transformers import SentenceTransformer

# Configuración de la página
st.set_page_config(page_title="Asistente IA para Asesoría Laboral", layout="wide")

# Título principal
st.title("Asistente IA para Asesoría Laboral")

# Función para extraer texto de PDFs
def extraer_texto_pdf(archivo_pdf):
    texto = ""
    documento = fitz.open(stream=archivo_pdf.read(), filetype="pdf")
    for pagina in documento:
        texto += pagina.get_text()
    return texto

# Función para dividir el texto en fragmentos manejables
def dividir_en_fragmentos(texto, tamano_max=1000):
    palabras = texto.split()
    fragmentos = []
    fragmento_actual = []
    
    for palabra in palabras:
        fragmento_actual.append(palabra)
        if len(" ".join(fragmento_actual)) >= tamano_max:
            fragmentos.append(" ".join(fragmento_actual))
            fragmento_actual = []
    
    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))
        
    return fragmentos

# Clase para la base de conocimiento
class BaseConocimiento:
    def __init__(self):
        # Cargar modelo de embeddings multilingüe (español incluido)
        self.modelo = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.fragmentos = []
        self.vectores = None
        
    def agregar_documento(self, texto, nombre_archivo, metadatos=None):
        # Dividir el texto
        fragmentos = dividir_en_fragmentos(texto)
        
        # Guardar fragmentos con metadatos
        for fragmento in fragmentos:
            self.fragmentos.append({
                'texto': fragmento,
                'fuente': nombre_archivo,
                'metadatos': metadatos
            })
        # Actualizar vectores
        self._actualizar_vectores()
        
    def _actualizar_vectores(self):
        # Crear vectores para todos los fragmentos
        if self.fragmentos:
            textos = [f['texto'] for f in self.fragmentos]
            self.vectores = self.modelo.encode(textos)
        
    def buscar(self, consulta, k=3):
        if not self.fragmentos or self.vectores is None:
            return []
            
        # Vectorizar la consulta
        vector_consulta = self.modelo.encode(consulta)
        
        # Calcular similitud con todos los fragmentos
        similitudes = np.dot(self.vectores, vector_consulta) / (
            np.linalg.norm(self.vectores, axis=1) * np.linalg.norm(vector_consulta)
        )
        
        # Encontrar los k más similares
        indices_top = np.argsort(-similitudes)[:k]
        
        # Devolver resultados
        resultados = []
        for idx in indices_top:
            resultados.append({
                'fragmento': self.fragmentos[idx],
                'similitud': similitudes[idx]
            })
        
        return resultados

# Inicializar la sesión si es necesario
if 'base_conocimiento' not in st.session_state:
    st.session_state.base_conocimiento = BaseConocimiento()

# Barra lateral para cargar documentos
with st.sidebar:
    st.header("Cargar Documentos")
    
    # Subir archivo
    uploaded_file = st.file_uploader("Selecciona un archivo PDF", type="pdf")
    
    if uploaded_file is not None:
        # Metadatos del documento
        titulo = st.text_input("Título del documento (opcional)")
        tema = st.text_input("Tema o categoría (opcional)")
        
        # Botón para procesar
        if st.button("Procesar Documento"):
            with st.spinner("Procesando documento..."):
                # Extraer texto
                texto = extraer_texto_pdf(uploaded_file)
                
                # Agregar a la base de conocimiento
                st.session_state.base_conocimiento.agregar_documento(
                    texto, 
                    uploaded_file.name,
                    {'titulo': titulo, 'tema': tema}
                )
                
                st.success(f"Documento '{uploaded_file.name}' procesado correctamente")
                st.text(f"Fragmentos: {len(st.session_state.base_conocimiento.fragmentos)}")

    # Mostrar documentos cargados
    if st.session_state.base_conocimiento.fragmentos:
        st.subheader("Documentos cargados")
        fuentes = set(f['fuente'] for f in st.session_state.base_conocimiento.fragmentos)
        for fuente in fuentes:
            st.write(f"- {fuente}")

# Área principal para consultas
st.header("Consulta sobre Asesoría Laboral")

# Campo de consulta
consulta_usuario = st.text_area("Escribe tu consulta sobre temas laborales", height=100)

# Botón para consultar
if st.button("Consultar"):
    if not consulta_usuario:
        st.warning("Por favor, escribe una consulta")
    elif not st.session_state.base_conocimiento.fragmentos:
        st.warning("No hay documentos cargados en la base de conocimiento")
    else:
        with st.spinner("Buscando respuesta..."):
            # Buscar información relevante
            resultados = st.session_state.base_conocimiento.buscar(consulta_usuario)
            
            if not resultados:
                st.error("No se encontró información relevante")
            else:
                # Mostrar respuesta
                st.subheader("Respuesta:")
                
                # Construir respuesta basada en los fragmentos recuperados
                respuesta = "Basado en la información disponible:\n\n"
                
                for i, res in enumerate(resultados, 1):
                    fragmento = res['fragmento']['texto']
                    fuente = res['fragmento']['fuente']
                    similitud = res['similitud']
                    
                    # Añadir a la respuesta general (versión simple)
                    respuesta += f"{fragmento}\n\n"
                
                st.write(respuesta)
                
                # Mostrar fuentes utilizadas
                st.subheader("Fuentes consultadas:")
                for res in resultados:
                    fuente = res['fragmento']['fuente']
                    metadata = res['fragmento']['metadatos']
                    titulo = metadata.get('titulo', 'Sin título') if metadata else 'Sin título'
                    
                    st.write(f"- {fuente} - {titulo}")
                
                # Expandible con fragmentos específicos
                with st.expander("Ver fragmentos específicos utilizados"):
                    for i, res in enumerate(resultados, 1):
                        st.markdown(f"**Fragmento {i}** (Similitud: {res['similitud']:.2f})")
                        st.text(res['fragmento']['texto'][:300] + "...")

# Sección para guardar/cargar la base de conocimiento
st.header("Gestionar Base de Conocimiento")

col1, col2 = st.columns(2)

with col1:
    if st.button("Guardar Base de Conocimiento"):
        if not st.session_state.base_conocimiento.fragmentos:
            st.warning("No hay datos para guardar")
        else:
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                pickle.dump(st.session_state.base_conocimiento, temp_file)
                temp_path = temp_file.name
            
            # Ofrecer descarga
            with open(temp_path, 'rb') as f:
                st.download_button(
                    label="Descargar Base de Conocimiento",
                    data=f,
                    file_name="base_conocimiento_laboral.pkl",
                    mime="application/octet-stream"
                )
            
            # Eliminar archivo temporal
            os.unlink(temp_path)

with col2:
    uploaded_base = st.file_uploader("Cargar Base de Conocimiento Guardada", type="pkl")
    if uploaded_base is not None:
        if st.button("Cargar Base"):
            try:
                st.session_state.base_conocimiento = pickle.load(uploaded_base)
                st.success("Base de conocimiento cargada correctamente")
            except Exception as e:
                st.error(f"Error al cargar la base: {e}")

# Información de uso
with st.expander("Cómo usar este asistente"):
    st.markdown("""
    **Instrucciones de uso:**
    
    1. **Cargar documentos**: Usa el panel lateral para subir documentos PDF relacionados con asesoría laboral.
    2. **Hacer consultas**: Escribe tu pregunta en el área de consulta y presiona "Consultar".
    3. **Gestionar la base**: Puedes guardar tu base de conocimiento para usarla después.
    
    **Recomendaciones:**
    - Carga documentos relevantes como leyes laborales, reglamentos, casos prácticos, etc.
    - Sé específico en tus consultas para obtener mejores resultados.
    - Esta es una versión básica que recupera información pero no genera respuestas elaboradas.
    """)

# Pie de página
st.markdown("---")
st.caption("Asistente IA para Asesorías Laborales - Versión 1.0")
