def leggi_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return f.readlines() 

def conta_righe(file):
    return len(file)

def conta_parole(file):
    pass

def parole_frequenti(file):
    pass