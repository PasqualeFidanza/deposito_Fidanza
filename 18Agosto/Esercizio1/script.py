def leggi_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return f.readlines() 

def conta_righe(file):
    return len(file)

def estrai_parole(file):
    parole = []
    for riga in file:
        parole.extend(riga.strip().split())
    return parole

def conta_parole(parole):
    return len(parole)

def parole_frequenti(parole):
    count_words = {}
    for parola in parole:
        p = parola.lower()
        count_words[p] = count_words.get(p, 0) + 1
    frequenti = sorted(count_words.items(), key=lambda x: x[1], reverse=True)
    return frequenti[:5]

path = r"C:\Users\VW778XM\OneDrive - EY\Documents\GitHub\deposito_Fidanza\18Agosto\Esercizio1\file.txt"

file = leggi_file(path)
print(conta_righe(file))
parole = estrai_parole(file)
frequenti = parole_frequenti(parole)
print(frequenti)

