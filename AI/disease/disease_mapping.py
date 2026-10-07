DISEASE_SPECIALTY_MAPPING = {
    "Migraine": "Thần kinh",
    "Vertigo": "Thần kinh",

    "Gastritis": "Tiêu hóa",
    "Gastroenteritis": "Tiêu hóa",

    "Pneumonia": "Hô hấp",
    "Asthma": "Hô hấp",

    "Hypertension": "Tim mạch",

    "Dermatitis": "Da liễu",
    "Psoriasis": "Da liễu",

    "Arthritis": "Cơ xương khớp",
}

def get_specialty(disease: str):
    return DISEASE_SPECIALTY_MAPPING.get(disease)