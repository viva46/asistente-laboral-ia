import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import os
import pickle
import tempfile
from sentence_transformers import SentenceTransformer

# OpenAI opcional
try:
    import openai
    openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
except ImportError:
    openai = None

# Configuración de página
st.set_page_config(page_title="Asistente IA para Asesoría Laboral", layout="wide")
st.title("Asistente IA para Asesoría Laboral")

# --- FUNCIONES Y CLASES ---

def extraer_texto_pdf(archivo_pdf):
    texto = ""
    documento = fitz.open(stream=archivo_pdf.read(), filetype="pdf")
    for pagina in documento:
        texto += pagina.get_text()
    return texto

def dividir_en_fragmentos(texto, tamano_max=1000):
    palabras = texto.split()
    fragmentos, actual = [], []

    for palabra in palabras:
        actual.append(palabra)
        if len(" ".join(actual)) >= tamano_max:
            fragmentos.append(" ".join(actual))
            actual = []

    if actual:
        fragmentos.append(" ".join(actual))
    return fragmentos

class BaseConocimiento:
    def __init__(self, modelo_path='paraphrase-multilingual-mpnet-base-v2'):
        self.modelo = SentenceTransformer(modelo_path)
        self.fragmentos = []
        self.vectores = None

    def agregar_documento(self, texto, nombre_archivo, metadatos=None):
        fragmentos = dividir_en_fragmentos(texto)
        nuevos_textos = []

        for fragmento in fragmentos:
            self.fragmentos.append({
                'texto': fragmento,
                'fuente': nombre_archivo,
                'metadatos': metadatos
            })
            nuevos_textos.append(fragmento)

        self._actualizar_vectores(nuevos_textos)

    def _actualizar_vectores(self, nuevos_textos=None):
        if nuevos_textos:
            nuevos_vectores = self.modelo.encode(nuevos_textos)
            if self.vectores is None:
                self.vectores = nuevos_vectores
            else:
                self.vectores = np.vstack([self.vectores, nuevos_vectores])

    def buscar(self, consulta, k=3):
        if not self.fragmentos or self.vectores is None:
            return []

        vector_consulta = self.modelo.encode(consulta)
        similitudes = np.dot(self.vectores, vector_consulta) / (
            np.linalg.norm(self.vectores, axis=1) * np.linalg.norm(vector_consulta)
        )

        indices_top = np.argsort(-similitudes)[:k]

        return [{
            'fragmento': self.fragmentos[i],
            'similitud': similitudes[i]
        } for i in indices_top]

    def exportar(self):
        return {
            "fragmentos": self.fragmentos,
            "vectores": self.vectores.tolist() if self.vectores is not None else None
        }

    def cargar(self, datos):
        self.fragmentos = datos.get("fragmentos", [])
        vectores = datos.get("vectores")
        if vectores is not None:
            self.vectores = np.array(vectores)

def generar_respuesta_llm(fragmentos, consulta):
    if not openai or not openai.api_key:
        return "⚠️ No se ha configurado OpenAI correctamente."

    contenido = "\n\n".join(fragmentos)
    prompt = f"""
    Contenido legal relevante:

    {contenido}

    Pregunta:
    {consulta}

    Responde con claridad y precisión.
    """

    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar respuesta: {e}"

# --- INICIALIZACIÓN DE SESIÓN ---
if 'base_conocimiento' not in st.session_state:
    st.session_state.base_conocimiento = BaseConocimiento()

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📄 Cargar Documentos")
    uploaded_file = st.file_uploader("Selecciona un PDF", type="pdf")

    if uploaded_file is not None:
        titulo = st.text_input("Título del documento (opcional)")
        tema = st.text_input("Tema o categoría (opcional)")

        if st.button("Procesar Documento"):
            with st.spinner("Procesando documento..."):
                try:
                    texto = extraer_texto_pdf(uploaded_file)
                    st.session_state.base_conocimiento.agregar_documento(
                        texto, uploaded_file.name, {'titulo': titulo, 'tema': tema}
                    )
                    st.success(f"Documento '{uploaded_file.name}' cargado.")
                except Exception as e:
                    st.error(f"Error al procesar el documento: {e}")

    # Mostrar fuentes cargadas
    if st.session_state.base_conocimiento.fragmentos:
        st.subheader("Documentos cargados")
        fuentes = list(set(f['fuente'] for f in st.session_state.base_conocimiento.fragmentos))
        for f in fuentes:
            st.write(f"- {f}")

# --- ÁREA PRINCIPAL ---
st.header("💬 Consulta sobre Asesoría Laboral")
consulta_usuario = st.text_area("Escribe tu pregunta:", height=100)

if st.button("Consultar"):
    if not consulta_usuario:
        st.warning("Escribe una consulta.")
    elif not st.session_state.base_conocimiento.fragmentos:
        st.warning("Primero debes cargar documentos.")
    else:
        with st.spinner("Buscando información..."):
            resultados = st.session_state.base_conocimiento.buscar(consulta_usuario)
            if not resultados:
                st.error("No se encontró información relevante.")
            else:
                st.subheader("📌 Fragmentos relevantes")
                for i, res in enumerate(resultados, 1):
                    st.markdown(f"**Fragmento {i}** (Similitud: {res['similitud']:.2f})")
                    st.text(res['fragmento']['texto'][:500] + "...")

                st.subheader("🧠 Respuesta generada por IA")
                textos = [res['fragmento']['texto'] for res in resultados]
                respuesta = generar_respuesta_llm(textos, consulta_usuario)
                st.write(respuesta)

# --- GESTIÓN DE BASE ---
st.header("💾 Gestionar Base de Conocimiento")
col1, col2 = st.columns(2)

with col1:
    if st.button("Guardar Base"):
        if not st.session_state.base_conocimiento.fragmentos:
            st.warning("No hay datos para guardar.")
        else:
            datos = st.session_state.base_conocimiento.exportar()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                pickle.dump(datos, temp_file)
                temp_path = temp_file.name

            with open(temp_path, 'rb') as f:
                st.download_button("Descargar Base", data=f, file_name="base_laboral.pkl")
            os.unlink(temp_path)

with col2:
    base_file = st.file_uploader("Cargar Base Guardada", type="pkl")
    if base_file and st.button("Cargar Base"):
        try:
            datos = pickle.load(base_file)
            bc = BaseConocimiento()
            bc.cargar(datos)
            st.session_state.base_conocimiento = bc
            st.success("Base cargada correctamente.")
        except Exception as e:
            st.error(f"Error al cargar: {e}")

# --- AYUDA ---
with st.expander("ℹ️ Cómo usar este asistente"):
    st.markdown("""
    **Instrucciones:**
    1. Sube documentos PDF laborales.
    2. Haz una consulta en lenguaje natural.
    3. El asistente te mostrará fragmentos relevantes y una respuesta automática.

    **Requiere clave de OpenAI para usar GPT-3.5.**
    """)
