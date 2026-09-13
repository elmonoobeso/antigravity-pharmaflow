import urllib.request
import urllib.error
import time

def ejecutar_benchmark_red(urls=None):
    """
    Ejecuta un benchmark de red básico midiendo la latencia a varios endpoints.
    Devuelve un diccionario con los resultados (URL -> latencia_ms) y un booleano
    indicando si la red en general está sana.
    """
    if urls is None:
        urls = [
            "https://1.1.1.1",
            "https://8.8.8.8",
            "https://www.google.com"
        ]
        
    resultados = []
    exitos = 0
    total = len(urls)
    
    for url in urls:
        inicio = time.time()
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=3):
                latencia = (time.time() - inicio) * 1000
                resultados.append({"url": url, "estado": "OK", "latencia_ms": round(latencia, 2)})
                exitos += 1
        except Exception as e:
            resultados.append({"url": url, "estado": f"Error: {str(e)}", "latencia_ms": None})
            
    sana = (exitos / total) >= 0.5 if total > 0 else False
    
    return {"resultados": resultados, "sana": sana, "exitos": exitos, "total": total}
