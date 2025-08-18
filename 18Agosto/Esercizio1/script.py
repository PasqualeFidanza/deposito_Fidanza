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
    frequenti = sorted(count_words, reverse=True)
    return frequenti[:5]




