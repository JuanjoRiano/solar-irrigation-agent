#Repite la consulta NASA para auditoría sin sobrescribir la copia usada.
import hashlib,json,urllib.request
from pathlib import Path
base=Path(__file__).resolve().parent
url=(base/'data/request_url.txt').read_text().strip()
with urllib.request.urlopen(url,timeout=120) as response:body=response.read()
json.loads(body) #Validar antes de escribir.
target=base/'data/nasa_power_redownload.json'
target.write_bytes(body)
print('Descarga:',target)
print('SHA-256:',hashlib.sha256(body).hexdigest())
print('NASA puede revisar sus productos. La copia original del análisis se conserva.')
