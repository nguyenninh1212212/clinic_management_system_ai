from typing import Dict


class QuestionGenerator:
    QUESTION_TEMPLATES: Dict[str, str] = {
        "headache": "Bạn có bị đau đầu không?",
        "dizziness": "Bạn có cảm thấy chóng mặt hoặc hoa mắt không?",
        "fever": "Bạn có bị sốt không?",
        "cough": "Bạn có bị ho không?",
        "sore_throat": "Bạn có bị đau hoặc rát họng không?",
        "nausea": "Bạn có cảm thấy buồn nôn không?",
        "vomiting": "Bạn có bị nôn hoặc ói không?",
        "chest_pain": "Bạn có bị đau hoặc tức ngực không?",
        "abdominal_pain": "Bạn có bị đau bụng không?",
        "shortness_of_breath": "Bạn có cảm thấy khó thở hoặc hụt hơi không?",
        "leg_pain": "Bạn có bị đau chân không?",
        "back_pain": "Bạn có bị đau lưng không?",
        "neck_pain": "Bạn có bị đau cổ không?",
        "fatigue": "Bạn có cảm thấy mệt mỏi hoặc uể oải không?",
        "rash": "Bạn có bị phát ban hoặc nổi mẩn không?",
    }

    def generate(self, symptom: str) -> str:
        question = self.QUESTION_TEMPLATES.get(symptom)

        if question is not None:
            return question

        return f"Bạn có gặp triệu chứng {symptom} không?"

    def generate_from_selection(self, selection: Dict) -> Dict:
        symptom = selection["symptom"]

        return {
            "symptom": symptom,
            "score": selection["score"],
            "question": self.generate(symptom),
        }
