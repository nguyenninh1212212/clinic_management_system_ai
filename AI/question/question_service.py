from typing import Optional

from AI.pipeline.symptom_pipeline import SymptomPipeline
from AI.disease.knowledge_base import DiseaseKnowledgeBase

from AI.question.answer_parser import AnswerParser
from AI.question.answer_parser_free_form import FreeFormAnswerParser
import math

QUESTION_TEMPLATES = {
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

    # =========================================================
    # MAIN FLOW
    # =========================================================

    def process_message(self, text: str) -> dict:
        text = text.strip()

        if not text:
            raise ValueError("Message cannot be empty")

        # -----------------------------------------------------
        # First message / no current question
        # -----------------------------------------------------

        if self.current_question is None:

            symptom_result = self.symptom_pipeline.process(text)

            ranked_diseases = self.rank_diseases()

            if self.should_stop(ranked_diseases):
                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    finished=True,
                )

            next_question = self.select_next_question(
                ranked_diseases
            )

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
            # ---------------------------------------------
            # Free-form answer
            # ---------------------------------------------

            symptom_result = self.symptom_pipeline.process(text)

            self._merge_current_question_context(
                text=text,
                current_symptom=current_symptom,
                symptom_result=symptom_result,
            )

        # Current question completed
        self.current_question = None

        self.question_count += 1

        # Re-rank after new information
        ranked_diseases = self.rank_diseases()

        # Stop condition
        if self.should_stop(ranked_diseases):
            return self._build_result(
                ranked_diseases=ranked_diseases,
                finished=True,
            )

        # Maximum question limit
        if self.question_count >= self.max_questions:
            return self._build_result(
                ranked_diseases=ranked_diseases,
                finished=True,
            )

        # Select next question
        next_question = self.select_next_question(
            ranked_diseases
        )

        return self._continue_question_flow(
            ranked_diseases,
            next_question,
        )

    # =========================================================
    # UPDATE PATIENT STATE
    # =========================================================

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

        # ---------------------------------------------
        # Determine current symptom value
        # ---------------------------------------------

        is_negated = self.answer_parser.is_negated(text)

        value = not is_negated

        # ---------------------------------------------
        # Extract duration / severity
        # ---------------------------------------------

        extra = self.free_form_parser.parse(text)

        duration = extra.get("duration")
        severity = extra.get("severity")

        # ---------------------------------------------
        # Update current symptom
        # ---------------------------------------------

        state_service.update_symptom(
            code=current_symptom,
            value=value,
            duration=duration,
            severity=severity,
        )

    # =========================================================
    # DISEASE RANKING
    # =========================================================

    def rank_diseases(self) -> list:

        state = self.symptom_pipeline.patient_state.get_state()

        patient_symptoms = {
            code
            for code, symptom_state in state.symptoms.items()
            if symptom_state.value is True
        }

        if not patient_symptoms:
            return []

        results = []

        for disease in self.knowledge_base.get_diseases():

            disease_symptoms = set(
                self.knowledge_base.get_symptoms(disease)
            )

            if not disease_symptoms:
                continue

            matched = patient_symptoms & disease_symptoms

            if not matched:
                continue

            patient_coverage = (
                len(matched) / len(patient_symptoms)
            )

            disease_coverage = (
                len(matched) / len(disease_symptoms)
            )

            score = (
                0.7 * patient_coverage
                + 0.3 * disease_coverage
            )

            results.append(
                {
                    "disease": disease,
                    "score": round(score, 4),
                    "matched_symptoms": sorted(matched),
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[: self.top_k]

    # =========================================================
    # QUESTION SELECTION
    # =========================================================

    def select_next_question(self, ranked_diseases: list) -> Optional[dict]:
        """
        Chọn symptom tiếp theo dựa trên Information Gain.

        Mỗi candidate symptom chia Top-K diseases thành:

            Có symptom
            Không có symptom

        Symptom nào làm giảm entropy của tập disease nhiều nhất
        sẽ được ưu tiên hỏi.
        """

        if not ranked_diseases:
            return None

        state = self.symptom_pipeline.patient_state.get_state()

        # TRUE hoặc FALSE đều được xem là đã biết.
        known_symptoms = {
            code
            for code, symptom_state in state.symptoms.items()
            if symptom_state.value is not None
        }

        # ---------------------------------------------------------
        # 1. Collect candidate symptoms
        # ---------------------------------------------------------

        candidates = set()

        for disease_result in ranked_diseases:
            disease = disease_result["disease"]

            for symptom in self.knowledge_base.get_symptoms(disease):
                if symptom in known_symptoms:
                    continue

                if symptom not in QUESTION_TEMPLATES:
                    continue

                candidates.add(symptom)

        if not candidates:
            return None

        # ---------------------------------------------------------
        # 2. Calculate current entropy
        # ---------------------------------------------------------

        current_entropy = self._calculate_disease_entropy(
            ranked_diseases
        )

        best_symptom = None
        best_gain = -1.0

        # ---------------------------------------------------------
        # 3. Evaluate each candidate
        # ---------------------------------------------------------

        for symptom in candidates:

            present_group = []
            absent_group = []

            for disease_result in ranked_diseases:
                disease = disease_result["disease"]

                disease_symptoms = set(
                    self.knowledge_base.get_symptoms(disease)
                )

                if symptom in disease_symptoms:
                    present_group.append(disease_result)
                else:
                    absent_group.append(disease_result)

            if not present_group or not absent_group:
                continue

            # -----------------------------------------------------
            # Expected entropy after asking this question
            # -----------------------------------------------------

            total = len(ranked_diseases)

            p_present = len(present_group) / total
            p_absent = len(absent_group) / total

            entropy_present = self._calculate_disease_entropy(
                present_group
            )

            entropy_absent = self._calculate_disease_entropy(
                absent_group
            )

            expected_entropy = (
                p_present * entropy_present
                + p_absent * entropy_absent
            )

            information_gain = (
                current_entropy - expected_entropy
            )

            # -----------------------------------------------------
            # 4. Ranking relevance
            # -----------------------------------------------------

            relevance = self._calculate_symptom_relevance(
                symptom=symptom,
                ranked_diseases=ranked_diseases,
            )

            # -----------------------------------------------------
            # 5. Final score
            # -----------------------------------------------------

            score = (
                0.7 * information_gain
                + 0.3 * relevance
            )

            if score > best_gain:
                best_gain = score
                best_symptom = symptom

        if best_symptom is None:
            return None

        return {
            "symptom": best_symptom,
            "question": self.generate_question(best_symptom),
            "score": round(best_gain, 4),
        }


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

    def generate_question(
        self,
        symptom: str,
    ) -> str:

        question = QUESTION_TEMPLATES.get(symptom)

        if question:
            return question

        return f"Bạn có triệu chứng {symptom} không?"

    # =========================================================
    # STOP CONDITION
    # =========================================================

    def should_stop(
        self,
        ranked_diseases: list,
    ) -> bool:

        if not ranked_diseases:
            return False

        top_score = ranked_diseases[0]["score"]

        return top_score >= 0.90

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