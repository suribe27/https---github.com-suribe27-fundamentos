import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

CARPETA_CVS = "hojas_de_vida"

STOP_WORDS = set([
    'el', 'la', 'de', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
    'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es', 'lo',
    'como', 'más', 'o', 'pero', 'sus', 'le', 'ya', 'fue', 'este',
    'ha', 'si', 'porque', 'esta', 'son', 'entre', 'cuando', 'muy',
    'sin', 'sobre', 'también', 'me', 'hasta', 'donde', 'quien', 'desde',
    'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros'
])

def extraer_texto_pdf(ruta):
    try:
        with open(ruta, 'rb') as f:
            lector = PyPDF2.PdfReader(f)
            texto = ''.join(pagina.extract_text() for pagina in lector.pages)
        return texto if len(texto.strip()) > 50 else None
    except Exception as e:
        print(f"   [!] Error en {os.path.basename(ruta)}: {e}")
        return None

def extraer_nombre(texto, archivo):
    match = re.search(r'nombre\s*(?:completo)?:\s*([a-záéíóúñ\s]+?)(?:\n|código|correo|teléfono|email)', texto.lower())
    if match:
        nombre = match.group(1).strip().title()
        nombre = re.sub(r'\s+', ' ', nombre)
        return nombre
    
    nombre = os.path.splitext(os.path.basename(archivo))[0]
    nombre = re.sub(r'^(cv|hoja|vida)_?', '', nombre, flags=re.I)
    nombre = nombre.replace('_', ' ').title()
    return nombre

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', ' ', texto)
    palabras = [p for p in texto.split() if len(p) > 2 and p not in STOP_WORDS]
    return ' '.join(palabras)

def analizar_candidatos(perfil_ideal, carpeta=CARPETA_CVS):
    print("=" * 85)
    print(" SISTEMA DE SELECCIÓN INTELIGENTE DE MONITORES ".center(85))
    print("=" * 85)
    print()
    
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print(f"[*] Carpeta '{carpeta}' creada. Coloca los PDFs ahí.\n")
        return None
    
    pdfs = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
    
    if not pdfs:
        print(f"[X] No hay archivos PDF en '{carpeta}'\n")
        return None
    
    print(f"[*] Carpeta: {carpeta}")
    print(f"[*] PDFs encontrados: {len(pdfs)}\n")
    
    print("[*] Leyendo archivos...")
    candidatos = []
    
    for i, pdf in enumerate(pdfs, 1):
        ruta = os.path.join(carpeta, pdf)
        print(f"   {i}. {pdf}...", end=" ")
        
        texto = extraer_texto_pdf(ruta)
        
        if texto:
            candidatos.append({
                'Nombre': extraer_nombre(texto, pdf),
                'Archivo': pdf,
                'Texto': limpiar_texto(texto)
            })
            print("[OK]")
        else:
            print("[ERROR]")
    
    if not candidatos:
        print("\n[X] No se pudo procesar ningún PDF\n")
        return None
    
    print(f"\n[OK] {len(candidatos)} CVs procesados correctamente\n")
    
    df = pd.DataFrame(candidatos)
    
    print("[*] Analizando con IA (TF-IDF + Similitud de Coseno)...")
    
    perfil_limpio = limpiar_texto(perfil_ideal)
    textos = df['Texto'].tolist() + [perfil_limpio]
    
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    matriz_tfidf = vectorizer.fit_transform(textos)
    
    similitudes = cosine_similarity(matriz_tfidf[-1], matriz_tfidf[:-1]).flatten()
    df['Score'] = similitudes
    
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    print(f"[OK] Análisis completado\n")
    
    print("=" * 85)
    print(" RANKING DE CANDIDATOS ".center(85))
    print("=" * 85)
    print()
    
    for i, row in df.iterrows():
        prefijo = ["[1]", "[2]", "[3]"][i] if i < 3 else f"[{i+1}]"
        barra = "█" * int(row['Score'] * 50)
        
        print(f"{prefijo} {row['Nombre']}")
        print(f"      Score: {row['Score']:.4f} ({int(row['Score']*100)}%) [{barra}]")
        print(f"      Archivo: {row['Archivo']}\n")
    
    print("=" * 85)
    print(f" RECOMENDADO: {df.iloc[0]['Nombre']} (Score: {df.iloc[0]['Score']:.4f}) ".center(85))
    print("=" * 85)
    print()
    
    df[['Nombre', 'Archivo', 'Score']].to_csv('ranking_monitores.csv', index=False)
    print("[*] Resultados guardados en: ranking_monitores.csv\n")
    
    return df

if __name__ == "__main__":
    
    perfil = """
    Buscamos estudiante para monitoría de Análisis de Datos con:
    - Dominio de Python (Pandas, NumPy, Matplotlib)
    - Conocimientos en estadística y análisis de datos
    - Experiencia previa en enseñanza, tutorías o monitorías
    - Excelente comunicación y paciencia
    - Promedio superior a 4.0
    - Capacidad para explicar conceptos complejos claramente
    """
    
    print("\n" + "╔" + "═" * 83 + "╗")
    print("║" + " SISTEMA DE SELECCIÓN INTELIGENTE DE MONITORES ".center(83) + "║")
    print("╚" + "═" * 83 + "╝")
    print()
    
    print("[*] Perfil ideal configurado")
    print("-" * 85)
    print(perfil.strip())
    print("-" * 85)
    print()
    
    input("Presiona ENTER para analizar las hojas de vida...")
    print()
    
    ranking = analizar_candidatos(perfil)
    
    if ranking is not None:
        print("[*] ESTADÍSTICAS")
        print("-" * 85)
        print(f"   • Total candidatos: {len(ranking)}")
        print(f"   • Score promedio: {ranking['Score'].mean():.4f}")
        print(f"   • Score máximo: {ranking['Score'].max():.4f}")
        print(f"   • Score mínimo: {ranking['Score'].min():.4f}")
        print("-" * 85)
        print()
        
        print("[*] SIGUIENTE PASO: Revisar los top 3 candidatos y hacer entrevistas\n")
    
    print("=" * 85)
    print()
    print("[*] INSTALACIÓN: pip install pandas scikit-learn PyPDF2")
    print("[*] USO: Coloca los PDFs en la carpeta 'hojas_de_vida/' y ejecuta")
    print()
    print("=" * 85)