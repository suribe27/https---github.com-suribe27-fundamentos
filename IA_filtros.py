"""
SISTEMA DE SELECCIÓN INTELIGENTE DE MONITORES
Lee hojas de vida PDF y las analiza con IA (TF-IDF + Cosine Similarity)
"""

import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# =============================================
# CONFIGURACIÓN
# =============================================

CARPETA_CVS = "hojas_de_vida"  # Carpeta con los PDFs

STOP_WORDS = set([
    'el', 'la', 'de', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
    'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es', 'lo',
    'como', 'más', 'o', 'pero', 'sus', 'le', 'ya', 'fue', 'este',
    'ha', 'si', 'porque', 'esta', 'son', 'entre', 'cuando', 'muy',
    'sin', 'sobre', 'también', 'me', 'hasta', 'donde', 'quien', 'desde',
    'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros'
])

# =============================================
# FUNCIONES CORE
# =============================================

def extraer_texto_pdf(ruta):
    """Extrae texto de PDF de forma eficiente"""
    try:
        with open(ruta, 'rb') as f:
            lector = PyPDF2.PdfReader(f)
            texto = ''.join(pagina.extract_text() for pagina in lector.pages)
        return texto if len(texto.strip()) > 50 else None
    except Exception as e:
        print(f"   ⚠️  Error en {os.path.basename(ruta)}: {e}")
        return None

def extraer_nombre(texto, archivo):
    """Extrae el nombre del estudiante del CV"""
    # Buscar "Nombre: Juan Pérez" o similar
    match = re.search(r'nombre\s*(?:completo)?:\s*([a-záéíóúñ\s]+)', texto.lower())
    if match:
        return match.group(1).strip().title()
    
    # Usar nombre del archivo
    nombre = os.path.splitext(os.path.basename(archivo))[0]
    return re.sub(r'^(cv|hoja|vida)_?', '', nombre, flags=re.I).replace('_', ' ').title()

def limpiar_texto(texto):
    """Limpia texto para análisis (optimizado)"""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', ' ', texto)
    palabras = [p for p in texto.split() if len(p) > 2 and p not in STOP_WORDS]
    return ' '.join(palabras)

# =============================================
# ANÁLISIS PRINCIPAL
# =============================================

def analizar_candidatos(perfil_ideal, carpeta=CARPETA_CVS):
    """
    Analiza todos los PDFs en la carpeta y genera ranking con IA
    
    Args:
        perfil_ideal (str): Descripción del monitor ideal (definido por el profesor)
        carpeta (str): Carpeta con los PDFs de las hojas de vida
    
    Returns:
        DataFrame con ranking ordenado por similitud
    """
    
    print("=" * 85)
    print(" 🤖 SISTEMA DE SELECCIÓN INTELIGENTE DE MONITORES ".center(85))
    print("=" * 85)
    print()
    
    # Verificar carpeta
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print(f"📁 Carpeta '{carpeta}' creada. Coloca los PDFs ahí.\n")
        return None
    
    # Obtener PDFs
    pdfs = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
    
    if not pdfs:
        print(f"❌ No hay archivos PDF en '{carpeta}'\n")
        return None
    
    print(f"📂 Carpeta: {carpeta}")
    print(f"📄 PDFs encontrados: {len(pdfs)}\n")
    
    # Leer PDFs
    print("📖 Leyendo archivos...")
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
            print("✅")
        else:
            print("❌")
    
    if not candidatos:
        print("\n❌ No se pudo procesar ningún PDF\n")
        return None
    
    print(f"\n✅ {len(candidatos)} CVs procesados correctamente\n")
    
    # Crear DataFrame
    df = pd.DataFrame(candidatos)
    
    # Análisis con IA
    print("🤖 Analizando con IA (TF-IDF + Similitud de Coseno)...")
    
    perfil_limpio = limpiar_texto(perfil_ideal)
    textos = df['Texto'].tolist() + [perfil_limpio]
    
    # Vectorización
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    matriz_tfidf = vectorizer.fit_transform(textos)
    
    # Calcular similitud
    similitudes = cosine_similarity(matriz_tfidf[-1], matriz_tfidf[:-1]).flatten()
    df['Score'] = similitudes
    
    # Ordenar por score
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    print(f"✅ Análisis completado\n")
    
    # Mostrar ranking
    print("=" * 85)
    print(" 🏆 RANKING DE CANDIDATOS ".center(85))
    print("=" * 85)
    print()
    
    for i, row in df.iterrows():
        emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        barra = "█" * int(row['Score'] * 50)
        
        print(f"{emoji} {row['Nombre']}")
        print(f"    Score: {row['Score']:.4f} ({int(row['Score']*100)}%) [{barra}]")
        print(f"    📁 {row['Archivo']}\n")
    
    print("=" * 85)
    print(f" ✅ RECOMENDADO: {df.iloc[0]['Nombre']} (Score: {df.iloc[0]['Score']:.4f}) ".center(85))
    print("=" * 85)
    print()
    
    # Guardar resultado
    df[['Nombre', 'Archivo', 'Score']].to_csv('ranking_monitores.csv', index=False)
    print("💾 Resultados guardados en: ranking_monitores.csv\n")
    
    return df

# =============================================
# EJECUCIÓN
# =============================================

if __name__ == "__main__":
    
    # PERFIL IDEAL - El profesor configura esto
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
    
    print("👨‍🏫 Perfil ideal configurado")
    print("-" * 85)
    print(perfil.strip())
    print("-" * 85)
    print()
    
    input("Presiona ENTER para analizar las hojas de vida...")
    print()
    
    # Analizar
    ranking = analizar_candidatos(perfil)
    
    if ranking is not None:
        print("📊 ESTADÍSTICAS")
        print("-" * 85)
        print(f"   • Total candidatos: {len(ranking)}")
        print(f"   • Score promedio: {ranking['Score'].mean():.4f}")
        print(f"   • Score máximo: {ranking['Score'].max():.4f}")
        print(f"   • Score mínimo: {ranking['Score'].min():.4f}")
        print("-" * 85)
        print()
        
        print("🎯 SIGUIENTE PASO: Revisar los top 3 candidatos y hacer entrevistas\n")
    
    print("=" * 85)
    print()
    print("📦 INSTALACIÓN: pip install pandas scikit-learn PyPDF2")
    print("📁 USO: Coloca los PDFs en la carpeta 'hojas_de_vida/' y ejecuta")
    print()
    print("=" * 85)