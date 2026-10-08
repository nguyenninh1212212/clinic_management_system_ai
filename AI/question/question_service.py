from typing import Optional

from AI.department.knowledge_base import DiseaseDepartmentKnowledgeBase
from AI.pipeline.symptom_pipeline import SymptomPipeline
from AI.disease.knowledge_base import DiseaseKnowledgeBase

from AI.question.answer_parser import AnswerParser
from AI.question.answer_parser_free_form import FreeFormAnswerParser

from AI.triage.triage_service import DepartmentTriageService
import math
from collections import Counter

QUESTION_TEMPLATES = {
    "headache": "Bạn có bị đau đầu không?",
    "dizziness": "Bạn có cảm thấy chóng mặt hoặc hoa mắt không?",
    "fever": "Bạn có bị sốt không?",
    "cough": "Bạn có bị ho không?",
    "sore_throat": "Bạn có bị đau hoặc rát họng không?",
    "nausea": "Bạn có cảm thấy buồn nôn không?",
    "vomiting": "Bạn có bị nôn hoặc ói không?",
    "chest_pain": "Bạn có bị đau hoặc tức ngực không?",
    "sharp_chest_pain": "Bạn có bị đau nhói ở ngực không?",
    "abdominal_pain": "Bạn có bị đau bụng không?",
    "shortness_of_breath": "Bạn có cảm thấy khó thở hoặc hụt hơi không?",
    "leg_pain": "Bạn có bị đau chân không?",
    "back_pain": "Bạn có bị đau lưng không?",
    "neck_pain": "Bạn có bị đau cổ không?",
    "fatigue": "Bạn có cảm thấy mệt mỏi hoặc uể oải không?",
    "rash": "Bạn có bị phát ban hoặc nổi mẩn không?",
    "wrist_pain": "Bạn có bị đau cổ tay không?",
    "seizures": "Bạn có từng bị co giật không?",
    "diminished_vision": "Bạn có bị giảm hoặc mờ thị lực không?",
    "pain_in_eye": "Bạn có bị đau mắt không?",
    "groin_mass": "Bạn có thấy khối bất thường ở vùng bẹn không?",

    "abnormal_involuntary_movements":
        "Bạn có gặp các cử động bất thường hoặc không kiểm soát được không?",

    "weakness":
        "Bạn có cảm thấy yếu tay hoặc chân không?",

    "arm_weakness":
        "Bạn có cảm thấy yếu ở cánh tay không?",

    "disturbance_of_memory":
        "Bạn có gặp vấn đề về trí nhớ không?",

    "decreased_appetite":
        "Bạn có bị giảm cảm giác thèm ăn không?",

    "difficulty_in_swallowing":
        "Bạn có gặp khó khăn khi nuốt thức ăn hoặc nước không?",

    "weight_gain":
        "Bạn có bị tăng cân bất thường không?",

    "symptoms_of_eye":
        "Bạn có gặp các triệu chứng bất thường về mắt không?",

    "shortness_of_breath":
        "Bạn có cảm thấy khó thở hoặc hụt hơi không?",

    "problems_with_movement": "Bạn có gặp khó khăn khi vận động hoặc cử động không?",
    "depressive_or_psychotic_symptoms": "Bạn có gặp các triệu chứng bất thường về tâm lý hoặc cảm xúc không?",
    "difficulty_speaking": "Bạn có gặp khó khăn khi nói hoặc diễn đạt lời nói không?",
    "focal_weakness": "Bạn có cảm thấy yếu rõ rệt ở một vùng tay hoặc chân không?",
    "slurring_words": "Bạn có bị nói khó hoặc nói không rõ lời không?",
    "vomiting": "Bạn có bị nôn hoặc ói không?",
    "blindness": "Bạn có bị mất thị lực không?",
    "nausea": "Bạn có cảm thấy buồn nôn không?",
}

class QuestionService:

    def __init__(
        self,
        model_path: str,
        knowledge_base_path: str,
        top_k: int = 5,
        max_questions: int = 8,
        
    ):
        self.symptom_pipeline = SymptomPipeline(model_path)

        self.knowledge_base = DiseaseKnowledgeBase(
            knowledge_base_path
        )

        self.answer_parser = AnswerParser()
        self.free_form_parser = FreeFormAnswerParser()

        self.top_k = top_k
        self.max_questions = max_questions

        self.current_question: Optional[dict] = None
        self.question_count = 0

        department_csv_path = (
            "training/dataset/department/disease_department.csv"
        )

        self.department_knowledge_base = (
            DiseaseDepartmentKnowledgeBase(
                department_csv_path
            )
        )

        self.department_triage = (
            DepartmentTriageService(
                self.department_knowledge_base
            )
        )

    # =========================================================
    # MAIN FLOW
    # =========================================================


    def process_message(self, text: str) -> dict:
        text = text.strip()

        if not text:
            raise ValueError("Message cannot be empty")
        
        if self.current_question is None:
            symptom_result = self.symptom_pipeline.process(text)

            ranked_diseases = self.rank_diseases()

            if self.should_stop(ranked_diseases):
                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    finished=True,
                )

            # Select next symptom using Information Gain
            next_symptom = self.select_best_question_symptom(
                ranked_diseases
            )

            if next_symptom is None:
                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    finished=True,
                )

            question = self.generate_question(
                next_symptom
            )

            next_question = {
                "symptom": next_symptom,
                "question": question,
            }

            return self._continue_question_flow(
                ranked_diseases,
                next_question,
            )

        # -----------------------------------------------------
        # Answering current question
        # -----------------------------------------------------

        current_symptom = self.current_question["symptom"]

        # First try simple YES / NO
        answer = self.answer_parser.parse_with_context(
            text,
            current_symptom,
        )

        if answer is not None:
            self._update_answer(
                symptom=current_symptom,
                value=answer,
            )

        else:
            # -------------------------------------------------
            # Free-form answer
            # -------------------------------------------------

            symptom_result = self.symptom_pipeline.process(text)

            self._merge_current_question_context(
                text=text,
                current_symptom=current_symptom,
                symptom_result=symptom_result,
            )

        # Current question completed
        self.current_question = None
        self.question_count += 1

        # -----------------------------------------------------
        # Re-rank after new information
        # -----------------------------------------------------

        ranked_diseases = self.rank_diseases()

        # -----------------------------------------------------
        # Stop condition
        # -----------------------------------------------------

        if self.should_stop(ranked_diseases):
            return self._build_result(
                ranked_diseases=ranked_diseases,
                finished=True,
            )

        # -----------------------------------------------------
        # Maximum question limit
        # -----------------------------------------------------

        if self.question_count >= self.max_questions:
            return self._build_result(
                ranked_diseases=ranked_diseases,
                finished=True,
            )

        # -----------------------------------------------------
        # Select next question using Information Gain
        # -----------------------------------------------------

        next_symptom = self.select_best_question_symptom(
            ranked_diseases
        )

        if next_symptom is None:
            return self._build_result(
                ranked_diseases=ranked_diseases,
                finished=True,
            )

        question = self.generate_question(
            next_symptom
        )

        next_question = {
            "symptom": next_symptom,
            "question": question,
        }

        return self._continue_question_flow(
            ranked_diseases,
            next_question,
        )


    def _update_answer(
        self,
        symptom: str,
        value: bool,
    ) -> None:

        state_service = self.symptom_pipeline.patient_state

        state_service.update_symptom(
            code=symptom,
            value=value,
        )
        
    def _merge_current_question_context(
        self,
        text: str,
        current_symptom: str,
        symptom_result: dict,
    ) -> None:
        state_service = self.symptom_pipeline.patient_state

        # -------------------------------------------------
        # Parse free-form answer
        # -------------------------------------------------

        extra = self.free_form_parser.parse(text)

        duration = extra.get("duration")
        severity = extra.get("severity")

        # -------------------------------------------------
        # Determine current symptom value
        # -------------------------------------------------

        is_negated = self.answer_parser.is_negated(text)

        value = not is_negated

        # -------------------------------------------------
        # Update current question symptom
        # -------------------------------------------------

        state_service.update_symptom(
            code=current_symptom,
            value=value,
            duration=duration,
            severity=severity,
        )

        # -------------------------------------------------
        # Update additional symptoms detected by NER
        # -------------------------------------------------

        symptoms = symptom_result.get("symptoms", [])

        for symptom in symptoms:
            if symptom == current_symptom:
                continue

            state_service.update_symptom(
                code=symptom,
                value=True,
            )
    # =========================================================
    # DISEASE RANKING
    # =========================================================

    def rank_diseases(self) -> list:
        state = self.symptom_pipeline.patient_state.get_state()

        positive_symptoms = set()
        negative_symptoms = set()

        for code, symptom_state in state.symptoms.items():
            if symptom_state.value is True:
                positive_symptoms.add(code)

            elif symptom_state.value is False:
                negative_symptoms.add(code)

        if not positive_symptoms:
            return []

        results = []

        for disease in self.knowledge_base.get_diseases():

            disease_symptoms = set(
                self.knowledge_base.get_symptoms(disease)
            )

            if not disease_symptoms:
                continue

            matched_positive = (
                positive_symptoms & disease_symptoms
            )

            matched_negative = (
                negative_symptoms & disease_symptoms
            )

            # Không có bằng chứng ủng hộ
            if not matched_positive:
                continue

            # --------------------------------
            # Positive evidence
            # --------------------------------

            total_positive_weight = sum(
                self.knowledge_base.get_symptom_weight(symptom)
                for symptom in positive_symptoms
            )

            matched_positive_weight = sum(
                self.knowledge_base.get_symptom_weight(symptom)
                for symptom in matched_positive
            )

            if total_positive_weight > 0:
                positive_score = (
                    matched_positive_weight
                    / total_positive_weight
                )
            else:
                positive_score = 0.0

            # --------------------------------
            # Disease coverage
            # --------------------------------

            disease_weight = sum(
                self.knowledge_base.get_symptom_weight(symptom)
                for symptom in disease_symptoms
            )

            if disease_weight > 0:
                disease_coverage = (
                    matched_positive_weight
                    / disease_weight
                )
            else:
                disease_coverage = 0.0

            # --------------------------------
            # Negative evidence
            # --------------------------------

            negative_weight = sum(
                self.knowledge_base.get_symptom_weight(symptom)
                for symptom in matched_negative
            )

            if disease_weight > 0:
                negative_penalty = (
                    negative_weight
                    / disease_weight
                )
            else:
                negative_penalty = 0.0

            # --------------------------------
            # Final score
            # --------------------------------
            score = (
                0.65 * positive_score
                + 0.35 * disease_coverage
                - 0.30 * negative_penalty
            )

            score = max(0.0, min(score, 1.0))

            results.append({
                "disease": disease,
                "score": round(score, 4),
                "matched_symptoms": sorted(matched_positive),
                "negative_symptoms": sorted(matched_negative),
                "missing_symptoms": sorted(
                    disease_symptoms
                    - positive_symptoms
                    - negative_symptoms
                ),
                "evidence": {
                    "positive_score": round(
                        positive_score,
                        4,
                    ),
                    "disease_coverage": round(
                        disease_coverage,
                        4,
                    ),
                    "negative_penalty": round(
                        negative_penalty,
                        4,
                    ),
                },
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:self.top_k]

    def _calculate_disease_entropy(
        self,
        ranked_diseases: list,
    ) -> float:
        """
        Entropy của phân phối disease dựa trên ranking score.
        """

        if not ranked_diseases:
            return 0.0

        total_score = sum(
            max(item["score"], 0.0)
            for item in ranked_diseases
        )

        if total_score <= 0:
            return 0.0

        entropy = 0.0

        for item in ranked_diseases:
            probability = (
                max(item["score"], 0.0)
                / total_score
            )

            if probability <= 0:
                continue

            entropy -= (
                probability
                * math.log2(probability)
            )

        return entropy


    def _calculate_symptom_relevance(
        self,
        symptom: str,
        ranked_diseases: list,
    ) -> float:
        """
        Tính mức độ liên quan của symptom với các disease
        đang đứng cao trong ranking.
        """

        if not ranked_diseases:
            return 0.0

        total_weight = 0.0
        matched_weight = 0.0

        for index, disease_result in enumerate(ranked_diseases):

            disease = disease_result["disease"]

            # Rank càng cao → weight càng lớn.
            rank_weight = 1.0 / (index + 1)

            disease_symptoms = set(
                self.knowledge_base.get_symptoms(disease)
            )

            total_weight += rank_weight

            if symptom in disease_symptoms:
                matched_weight += rank_weight

        if total_weight == 0:
            return 0.0

        return matched_weight / total_weight

    def _binary_entropy(self, probability: float) -> float:

        if probability <= 0 or probability >= 1:
            return 0.0

        return -(
        probability * math.log2(probability)
        + (1 - probability) * math.log2(1 - probability)
    )


    def _question_discrimination_score(
    self,
    symptom: str,
    ranked_diseases: list,
    information_gain: float,
) -> float:
   

        ranking_weight = 0.0
        total_weight = 0.0

        for index, disease_result in enumerate(ranked_diseases):
            disease = disease_result["disease"]

        # Bệnh đứng đầu có trọng số cao hơn.
            weight = 1.0 / (index + 1)

            disease_symptoms = set(
                self.knowledge_base.get_symptoms(disease)
            )

            if symptom in disease_symptoms:
                ranking_weight += weight

            total_weight += weight
    
        if total_weight == 0:
            return information_gain

        relevance = ranking_weight / total_weight

    # 70% khả năng phân biệt
    # 30% relevance với các disease đang ranking cao
        return (
        0.7 * information_gain
        + 0.3 * relevance
    )


    # =========================================================
    # QUESTION GENERATION
    # =========================================================
    def generate_question(self, symptom: str) -> str:
        question = QUESTION_TEMPLATES.get(symptom)

        if question:
            return question

        readable_symptom = symptom.replace("_", " ")

        return (
            f"Bạn có triệu chứng "
            f"{readable_symptom} không?"
        )
    # =========================================================
    # STOP CONDITION
    # =========================================================

    def should_stop(self, ranked_diseases: list) -> bool:
        if not ranked_diseases:
            return False

        # Phải có đủ một số lượng câu hỏi tối thiểu
        if self.question_count < 3:
            return False

        top_score = ranked_diseases[0]["score"]

        if top_score < 0.90:
            return False

        if len(ranked_diseases) == 1:
            return True

        second_score = ranked_diseases[1]["score"]

        margin = top_score - second_score

        return margin >= 0.15

    # =========================================================
    # CONTINUE FLOW
    # =========================================================

    def _continue_question_flow(
        self,
        ranked_diseases: list,
        next_question: Optional[dict],
    ) -> dict:

        if next_question is None:
            return self._build_result(
                ranked_diseases=ranked_diseases,
                finished=True,
            )

        self.current_question = next_question

        return self._build_result(
            ranked_diseases=ranked_diseases,
            next_question=next_question,
            finished=False,
        )

    # =========================================================
    # RESULT
    # =========================================================

    def _build_result(
        self,
        ranked_diseases: list,
        next_question: Optional[dict] = None,
        finished: bool = False,
    ) -> dict:

        state = self.symptom_pipeline.patient_state.get_state()
        department_triage = self.suggest_department(
            ranked_diseases
        )
        symptoms = []

        for code, symptom_state in state.symptoms.items():

            if symptom_state.value is True:
                symptoms.append(code)

        return {
            "question": (
                next_question["question"]
                if next_question
                else None
            ),
            "symptoms": symptoms,
            "patient_state": state,
            "ranked_diseases": ranked_diseases,
            "department_triage": department_triage,
            "next_question": next_question,
            "finished": finished,
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:

        self.symptom_pipeline.patient_state.clear()

        self.current_question = None
        self.question_count = 0

    def select_best_question_symptom(
    self,
    ranked_diseases: list,
    ) -> str | None:

        if not ranked_diseases:
            return None

        state = self.symptom_pipeline.patient_state.get_state()

        known_symptoms = {
            code
            for code, symptom_state in state.symptoms.items()
            if symptom_state.value is not None
        }

        diseases = [
            item["disease"]
            for item in ranked_diseases
        ]

        candidates = set()

        for disease in diseases:

            disease_symptoms = (
                self.knowledge_base.get_symptoms(disease)
            )

            for symptom in disease_symptoms:

                if symptom in known_symptoms:
                    continue

                candidates.add(symptom)

        if not candidates:
            return None

        best_symptom = None
        best_gain = -1.0

        for symptom in candidates:

            gain = self._question_information_gain(
                symptom=symptom,
                diseases=diseases,
            )

            if gain > best_gain:

                best_gain = gain
                best_symptom = symptom

        return best_symptom

    def _entropy(self, diseases: list[str]) -> float:

        if not diseases:
            return 0.0

        counts = Counter(diseases)

        total = len(diseases)

        entropy = 0.0

        for count in counts.values():

            probability = count / total

            entropy -= (
                probability
                * math.log2(probability)
            )

        return entropy

    def _question_information_gain(
        self,
        symptom: str,
        diseases: list[str],
    ) -> float:

        if not diseases:
            return 0.0

        yes_group = []
        no_group = []

        for disease in diseases:

            disease_symptoms = (
                self.knowledge_base.get_symptoms(disease)
            )

            if symptom in disease_symptoms:
                yes_group.append(disease)
            else:
                no_group.append(disease)

        total = len(diseases)

        if not yes_group or not no_group:
            return 0.0

        current_entropy = self._entropy(diseases)

        yes_weight = len(yes_group) / total
        no_weight = len(no_group) / total

        conditional_entropy = (
            yes_weight * self._entropy(yes_group)
            + no_weight * self._entropy(no_group)
        )

        return current_entropy - conditional_entropy

    def suggest_department(
        self,
        ranked_diseases: list,
    ) -> dict:
        return self.department_triage.suggest_department(
            ranked_diseases
        )