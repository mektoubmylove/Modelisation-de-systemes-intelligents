# Extraction sexe et année depuis le nom du fichier
def extract_metadata(filename):
    matches = re.findall(r'\(([^)]*)\)', filename)
    if len(matches) >= 5:
        sexe_code = matches[3]
        year = matches[4]

        if sexe_code == '1':
            sexe = 'homme'
        elif sexe_code == '2':
            sexe = 'femme'
        else:
            return None

        try:
            year = int(year)
        except:
            return None

        return sexe, year
    return None


# Prédiction du modèle sur un texte
def predict_sexe(text):
    model.eval()
    X = tfidf.transform([text])
    X_tensor = torch.FloatTensor(X.toarray()).to(device)

    with torch.no_grad():
        output = model(X_tensor)
        pred_class = output.argmax(1).item()

    return le.inverse_transform([pred_class])[0]

test_meta = []

for idx, row in test_df.iterrows():
    path = row["path"]
    text = row["text"]

    filename = os.path.basename(path)
    meta = extract_metadata(filename)

    if meta:
        true_sexe, year = meta
        pred_sexe = predict_sexe(text)

        # filtrer uniquement les textes bien prédits
        if pred_sexe == true_sexe:
            test_meta.append({
                "index": idx,
                "path": path,
                "sexe": true_sexe,
                "year": year
            })

print(f"Textes conservés (bien prédits) : {len(test_meta)} / {len(test_df)}")


# Fonction pour créer des sous-ensembles de 10 textes
def create_subsets(data, window=30):
    groups = defaultdict(list)

    for item in data:
        window_start = (item["year"] // window) * window
        key = (item["sexe"], window_start)
        groups[key].append(item)

    subsets = []
    for (sexe, start_year), items in groups.items():
        if len(items) >= 10:
            items = sorted(items, key=lambda x: x["year"])

            for i in range(0, len(items) - 9, 10):
                subset = items[i:i + 10]
                subsets.append({
                    "sexe": sexe,
                    "periode": f"{start_year}-{start_year + window}",
                    "indices": [x["index"] for x in subset],
                    "paths": [x["path"] for x in subset]
                })

    return subsets


# Création des sous-ensembles 
all_subsets = create_subsets(test_meta)


print(f"\nSous-ensembles créés : {len(all_subsets)}")
for i, s in enumerate(all_subsets):
    print(f"\nSous-ensemble {i} | Sexe: {s['sexe']} | Période: {s['periode']} | Indices: {s['indices']}")

"""
Textes conservés (bien prédits) : 253 / 285

Sous-ensembles créés : 21

Sous-ensemble 0 | Sexe: femme | Période: 2010-2040 | Indices: [0, 194, 235, 238, 280, 1, 146, 150, 158, 201]

Sous-ensemble 1 | Sexe: femme | Période: 2010-2040 | Indices: [203, 52, 63, 102, 103, 197, 252, 268, 2, 65]

Sous-ensemble 2 | Sexe: femme | Période: 2010-2040 | Indices: [185, 237, 277, 108, 145, 210, 214, 243, 11, 84]

Sous-ensemble 3 | Sexe: femme | Période: 2010-2040 | Indices: [105, 116, 178, 200, 212, 265, 38, 106, 109, 114]

Sous-ensemble 4 | Sexe: femme | Période: 2010-2040 | Indices: [168, 211, 228, 251, 8, 19, 59, 100, 151, 163]

Sous-ensemble 5 | Sexe: femme | Période: 2010-2040 | Indices: [180, 181, 204, 215, 227, 74, 126, 144, 160, 183]

Sous-ensemble 6 | Sexe: femme | Période: 2010-2040 | Indices: [257, 57, 124, 136, 256, 56, 94, 92, 244, 127]

Sous-ensemble 7 | Sexe: femme | Période: 1980-2010 | Indices: [271, 96, 177, 282, 189, 75, 12, 51, 241, 35]

Sous-ensemble 8 | Sexe: femme | Période: 1980-2010 | Indices: [80, 47, 69, 156, 263, 68, 142, 149, 172, 217]

Sous-ensemble 9 | Sexe: femme | Période: 1980-2010 | Indices: [240, 258, 28, 42, 224, 225, 275, 4, 91, 34]

Sous-ensemble 10 | Sexe: femme | Période: 1980-2010 | Indices: [58, 72, 216, 232, 88, 283, 107, 269, 133, 266]

Sous-ensemble 11 | Sexe: femme | Période: 1980-2010 | Indices: [220, 245, 49, 138, 193, 229, 3, 10, 24, 208]

Sous-ensemble 12 | Sexe: homme | Période: 1980-2010 | Indices: [32, 259, 20, 176, 169, 167, 62, 284, 33, 157]

Sous-ensemble 13 | Sexe: homme | Période: 1980-2010 | Indices: [221, 246, 199, 218, 147, 165, 46, 61, 274, 6]

Sous-ensemble 14 | Sexe: homme | Période: 1980-2010 | Indices: [31, 175, 187, 236, 186, 29, 121, 148, 76, 22]

Sous-ensemble 15 | Sexe: homme | Période: 1920-1950 | Indices: [18, 9, 41, 141, 111, 14, 83, 166, 53, 130]

Sous-ensemble 16 | Sexe: homme | Période: 1920-1950 | Indices: [230, 267, 36, 188, 279, 248, 132, 135, 247, 78]

Sous-ensemble 17 | Sexe: homme | Période: 1950-1980 | Indices: [86, 152, 223, 27, 273, 87, 162, 39, 48, 128]

Sous-ensemble 18 | Sexe: homme | Période: 1950-1980 | Indices: [98, 45, 173, 255, 119, 73, 190, 77, 93, 113]

Sous-ensemble 19 | Sexe: homme | Période: 1890-1920 | Indices: [25, 30, 202, 55, 231, 17, 54, 139, 272, 164]

Sous-ensemble 20 | Sexe: femme | Période: 1950-1980 | Indices: [249, 171, 153, 184, 253, 207, 261, 26, 115, 242]
"""