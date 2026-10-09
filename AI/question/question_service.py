from typing import Optional
from pathlib import Path
from AI.department.knowledge_base import DiseaseDepartmentKnowledgeBase
from AI.pipeline.symptom_pipeline import NERPredictor, SymptomPipeline
from AI.disease.knowledge_base import DiseaseKnowledgeBase

from AI.question.answer_parser import AnswerParser
from AI.question.answer_parser_free_form import FreeFormAnswerParser
from AI.question.confirmation_parser import ConfirmationParser
from AI.question.confirmation_knowledge_base import (
    ConfirmationKnowledgeBase,
)
from AI.question.question_knowledge_base import (
    QuestionKnowledgeBase,
)
from AI.triage.triage_service import DepartmentTriageService
from AI.triage.safety_triage import SafetyTriage
import math
from collections import Counter
from AI.question.clarification_knowledge_base import (
    ClarificationKnowledgeBase,
)
class QuestionService:

    def __init__(
        self,
        model_path: str | Path,
        knowledge_base_path: str | Path,
        top_k: int = 5,
        max_questions: int = 8,
        ner: NERPredictor | None = None,

    ):
        self.symptom_pipeline = SymptomPipeline(
            model_path,
            ner=ner,
        )

        self.knowledge_base = DiseaseKnowledgeBase(
            knowledge_base_path
        )
        self.awaiting_department_confirmation = False
        self.answer_parser = AnswerParser()
        self.free_form_parser = FreeFormAnswerParser()

        BASE_DIR = Path(__file__).resolve().parents[2]

        confirmation_csv_path = (
            BASE_DIR
            / "training"
            / "dataset"
            / "question"
            / "confirmation_questions.csv"
        )

        self.confirmation_knowledge_base = (
            ConfirmationKnowledgeBase(
                confirmation_csv_path
            )
        )

        self.confirmation_parser = ConfirmationParser(
            self.confirmation_knowledge_base
        )

        self.awaiting_confirmation = False


        BASE_DIR = Path(__file__).resolve().parents[2]

        question_csv_path = (
            BASE_DIR
            / "training"
            / "dataset"
            / "question"
            / "symptom_question_final.csv"
        )

        self.question_knowledge_base = (
            QuestionKnowledgeBase(
                question_csv_path
            )
        )

        clarification_csv_path = (
            BASE_DIR
            / "training"
            / "dataset"
            / "question"
            / "clarification_questions.csv"
        )

        self.clarification_knowledge_base = (
            ClarificationKnowledgeBase(
                clarification_csv_path
            )
        )

        self.top_k = top_k
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1")
        self.max_questions = max_questions

        self.current_question: Optional[dict] = None
        self.question_count = 0
        self.questions_asked = 0
        self.symptom_details_asked: set[str] = set()
        self.awaiting_confirmation = False

        department_csv_path = (
            BASE_DIR
            / "training"
            / "dataset"
            / "department"
            / "disease_department.csv"
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
        self.safety_triage = SafetyTriage()

    # =========================================================
    # MAIN FLOW
    # =========================================================


    def process_message(self, text: str) -> dict:

        text = text.strip()

        if not text:
            raise ValueError(
                "Message cannot be empty"
            )

        safety_assessment = self.safety_triage.assess(text)
        if safety_assessment is not None:
            self._store_symptoms_with_polarity(
                text,
                self.symptom_pipeline.normalizer.find_in_text(text),
            )
            self.current_question = None
            self.awaiting_confirmation = False
            self.awaiting_department_confirmation = False
            result = self._build_result(
                ranked_diseases=self.rank_diseases(),
                next_question=None,
                finished=False,
            )
            result.update(
                {
                    "status": "emergency",
                    "finished": False,
                    "message": safety_assessment["message"],
                    "safety": safety_assessment,
                    "question": None,
                    "next_question": None,
                }
            )
            return result

        if self._is_end_intent(text):
            self.current_question = None
            self.awaiting_confirmation = False
            self.awaiting_department_confirmation = False

            result = self._build_result(
                ranked_diseases=self.rank_diseases(),
                next_question=None,
                finished=True,
            )

            result["status"] = "ended"
            result["finished"] = True
            result["question"] = None
            result["next_question"] = None

            return result
        # ==================================================
        # 1. ĐANG CHỜ USER CONFIRM
        # ==================================================

        if self.awaiting_department_confirmation:
            return self._handle_department_confirmation(text)


        if self.awaiting_confirmation:

            return self._handle_confirmation(
                text
            )

        # ==================================================
        # 2. ĐANG CHỜ USER TRẢ LỜI CÂU HỎI
        # ==================================================

        if self.current_question is not None:

            current_symptom = (
                self.current_question["symptom"]
            )

            if self.current_question.get("question_type") == "symptom_details":
                extracted_symptoms = self._record_free_form_symptoms(text)
                extra = self.free_form_parser.parse(text)
                self.symptom_pipeline.patient_state.update_symptom(
                    code=current_symptom,
                    duration=extra.get("duration"),
                    severity=extra.get("severity"),
                )
                value = (
                    True
                    if extra.get("duration")
                    or extra.get("severity")
                    else None
                )
            else:
                symptom_result = self.symptom_pipeline.process(
                    text,
                    update_state=False,
                )
                extracted_symptoms = symptom_result.get("symptoms", [])
                value = self._merge_current_question_context(
                    text=text,
                    current_symptom=current_symptom,
                    symptom_result=symptom_result,
                )

            if value is not None:
                self.question_count += 1
            elif (
                not extracted_symptoms
                or self.current_question.get("question_type")
                == "symptom_details"
            ):
                if self.current_question.get("question_type") == "symptom_details":
                    clarification_question = (
                        "Bạn cho biết thêm triệu chứng bắt đầu từ khi nào "
                        "hoặc mức độ ra sao nhé."
                    )
                    question_type = "symptom_details"
                else:
                    clarification_question = (
                        "Bạn có thể cho biết rõ hơn không? "
                        "Triệu chứng này có xuất hiện không, "
                        "hay bạn hoàn toàn không gặp triệu chứng đó?"
                    )
                    question_type = "unclear_symptom"

                return self._build_result(
                    ranked_diseases=self.rank_diseases(),
                    next_question={
                        "symptom": current_symptom,
                        "question_type": question_type,
                        "question": clarification_question,
                    },
                    finished=False,
                )

            self.current_question = None
            # self.question_count += 1

            ranked_diseases = (
                self.rank_diseases()
            )

            detail_question = self._next_symptom_details_question()
            if detail_question is not None:
                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    next_question=detail_question,
                    finished=False,
                )

            # ==================================================
            # ĐỦ THÔNG TIN → CONFIRMATION
            # ==================================================

            if self.should_stop(
                ranked_diseases
            ):

                self.awaiting_confirmation = True

                confirmation_question = (
                    self.get_confirmation_question()
                )

                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    next_question={
                        "symptom": None,
                        "question_type": "confirmation",
                        "question": (
                            confirmation_question
                        ),
                    },
                    finished=False,
                )

            # ==================================================
            # CHƯA ĐỦ → HỎI TIẾP
            # ==================================================

            next_symptom = (
                self.select_best_question_symptom(
                    ranked_diseases
                )
            )

            if next_symptom is None:

                confirmation_question = (
                    self.get_confirmation_question()
                )

                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    next_question={
                        "symptom": None,
                        "question_type": "confirmation",
                        "question": (
                            confirmation_question
                        ),
                    },
                    finished=False,
                )

            question = self.generate_question(
                next_symptom
            )

            self.current_question = {
                "symptom": next_symptom,
                "question_type": "yes_no",
                "question": question,
            }

            return self._build_result(
                ranked_diseases=ranked_diseases,
                next_question=self.current_question,
                finished=False,
            )

        # ==================================================
        # 3. MESSAGE ĐẦU TIÊN / FREE-FORM
        # ==================================================

        return self._continue_from_free_form(
            text
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
    ) -> Optional[bool]:
        state_service = self.symptom_pipeline.patient_state

        extra = self.free_form_parser.parse(text)
        symptoms = symptom_result.get("symptoms", [])
        symptom_codes = [
            symptom["code"] if isinstance(symptom, dict) else symptom
            for symptom in symptoms
        ]
        all_codes = list(dict.fromkeys(
            [*symptom_codes, current_symptom]
        ))
        polarity = self.free_form_parser.parse_symptom_polarity(
            text=text,
            symptoms=all_codes,
        )

        for code in symptom_codes:
            value = polarity.get(code, True)
            state_service.update_symptom(
                code=code,
                value=value,
                duration=extra.get("duration") if value else None,
                severity=extra.get("severity") if value else None,
            )

        current_value = polarity.get(current_symptom)
        if current_value is None:
            current_value = self.answer_parser.parse_with_context(
                text=text,
                symptom=current_symptom,
            )

        if current_value is not None:
            state_service.update_symptom(
                code=current_symptom,
                value=current_value,
                duration=extra.get("duration"),
                severity=extra.get("severity"),
            )

        return current_value
    # =========================================================
    # DISEASE RANKING
    # =========================================================

    def rank_diseases(self) -> list:
        """Return heuristic disease scores, not calibrated probabilities."""
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
            key=lambda item: (
                -item["score"],
                item["disease"],
            ),
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
        question = (
            self.question_knowledge_base
            .get_question_text(symptom)
        )

        if question:
            return question

        return (
            f"Chưa có câu hỏi cho symptom: "
            f"{symptom}"
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
        question_type = (
            next_question.get("question_type")
            if next_question
            else None
        )
        terminal_question_types = {
            "confirmation",
            "department_confirmation",
        }

        if (
            not finished
            and next_question
            and question_type not in terminal_question_types
            and self.questions_asked >= self.max_questions
        ):
            self.current_question = None
            self.awaiting_confirmation = True
            self.awaiting_department_confirmation = False
            next_question = {
                "symptom": None,
                "question_type": "confirmation",
                "question": self.get_confirmation_question(),
            }
            question_type = "confirmation"
        elif (
            not finished
            and next_question
            and question_type not in terminal_question_types
        ):
            self.questions_asked += 1

        if question_type == "confirmation":
            self.awaiting_confirmation = True
            self.awaiting_department_confirmation = False
        elif question_type == "department_confirmation":
            self.awaiting_confirmation = False
            self.awaiting_department_confirmation = True

        state = (
            self.symptom_pipeline
            .patient_state
            .get_state()
        )

        department_triage = (
            self.suggest_department(
                ranked_diseases
            )
        )

        symptoms = []

        for code, symptom_state in (
            state.symptoms.items()
        ):
            if symptom_state.value is True:
                symptoms.append(code)

        if finished:
            status = "completed"

        elif self.awaiting_confirmation:
            status = "awaiting_confirmation"

        elif (
            next_question
            and next_question.get(
                "question_type"
            ) == "unclear_symptom"
        ):
            status = "needs_clarification"
        elif self.awaiting_department_confirmation:
            status = "awaiting_department_confirmation"
        elif self.awaiting_confirmation:
            status = "awaiting_confirmation"
        else:
            status = "asking_question"

        return {
            "status": status,
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
            "questions_asked": self.questions_asked,
        }

    def _question_limit_result(self) -> dict:
        self.current_question = None
        self.awaiting_confirmation = False
        self.awaiting_department_confirmation = False
        result = self._build_result(
            ranked_diseases=self.rank_diseases(),
            next_question=None,
            finished=False,
        )
        result["status"] = "question_limit_reached"
        result["finished"] = False
        result["message"] = (
            "Đã đạt giới hạn câu hỏi tự động. "
            "Thông tin hiện có được giữ lại để tham khảo."
        )
        return result

    def _record_free_form_symptoms(self, text: str) -> list[dict]:
        symptom_result = self.symptom_pipeline.process(
            text,
            update_state=False,
        )
        extracted_symptoms = symptom_result.get("symptoms", [])
        self._store_symptoms_with_polarity(
            text,
            extracted_symptoms,
        )
        return extracted_symptoms

    def _store_symptoms_with_polarity(
        self,
        text: str,
        extracted_symptoms: list[dict],
    ) -> None:
        extra = self.free_form_parser.parse(text)
        symptom_codes = [
            symptom["code"]
            for symptom in extracted_symptoms
        ]
        polarity = self.free_form_parser.parse_symptom_polarity(
            text=text,
            symptoms=symptom_codes,
        )
        for symptom in extracted_symptoms:
            code = symptom["code"]
            self.symptom_pipeline.patient_state.update_symptom(
                code=code,
                value=polarity.get(code, True),
                duration=extra.get("duration"),
                severity=extra.get("severity"),
            )

    def _next_symptom_details_question(self) -> dict | None:
        state = self.symptom_pipeline.patient_state.get_state()
        positive_symptoms = [
            (code, symptom_state)
            for code, symptom_state in state.symptoms.items()
            if symptom_state.value is True
        ]
        if len(positive_symptoms) != 1:
            return None

        code, symptom_state = positive_symptoms[0]
        if (
            code in self.symptom_details_asked
            or (
                symptom_state.duration is not None
                and symptom_state.severity is not None
            )
        ):
            return None

        self.symptom_details_asked.add(code)
        self.current_question = {
            "symptom": code,
            "question_type": "symptom_details",
            "question": (
                "Để hiểu rõ hơn về triệu chứng này, bạn cho biết "
                "nó bắt đầu từ khi nào và mức độ ra sao?"
            ),
        }
        return self.current_question


    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:

        self.symptom_pipeline.patient_state.clear()

        self.current_question = None

        self.question_count = 0
        self.questions_asked = 0
        self.symptom_details_asked.clear()

        self.awaiting_confirmation = False

        self.awaiting_department_confirmation = False


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

        # Chỉ chọn symptom có câu hỏi trong QuestionKnowledgeBase.
        available_symptoms = set(
            self.question_knowledge_base.get_question_symptoms()
        )

        diseases = [
            item["disease"]
            for item in ranked_diseases
        ]

        candidate_set = set()

        for disease in diseases:
            disease_symptoms = (
                self.knowledge_base.get_symptoms(disease)
            )

            for symptom in disease_symptoms:
                if symptom in known_symptoms:
                    continue

                if symptom not in available_symptoms:
                    continue

                candidate_set.add(symptom)

        candidates = [
            symptom
            for symptom in self.question_knowledge_base.get_question_symptoms()
            if symptom in candidate_set
        ]
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

    # def select_fallback_question_symptom(
    #     self,
    # ) -> str | None:
    #     state = (
    #         self.symptom_pipeline
    #         .patient_state
    #         .get_state()
    #     )

    #     known_symptoms = {
    #         code
    #         for code, symptom_state in state.symptoms.items()
    #         if symptom_state.value is not None
    #     }

    #     symptoms = (
    #         self.question_knowledge_base
    #         .get_question_symptoms()
    #     )

    #     for symptom in symptoms:
    #         if symptom not in known_symptoms:
    #             return symptom

    #     return None

    def get_clarification_question(
        self,
        question_type: str = "unclear_symptom",
    ) -> str:

        question = (
            self.clarification_knowledge_base
            .get_question(question_type)
        )

        if question:
            return question

        return (
            "Bạn có thể cung cấp thêm thông tin "
            "về triệu chứng của mình không?"
        )

    def get_confirmation_question(self) -> str:

        question = (
            self.confirmation_knowledge_base
            .get_question()
        )

        if question:
            return question

        return (
            "Bạn có muốn bổ sung thêm "
            "thông tin về triệu chứng không?"
        )

    def _handle_confirmation(
        self,
        text: str,
    ) -> dict:

        result = self.confirmation_parser.parse(
            text
        )

        confirmation_value = result["value"]


        if confirmation_value is True:
            self.awaiting_confirmation = False

            ranked_diseases = self.rank_diseases()
            department_result = self.suggest_department(
                ranked_diseases
            )

            department = department_result.get(
                "suggested_department"
            )

            if not department:
                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    next_question={
                        "symptom": None,
                        "question_type": "unclear_symptom",
                        "question": (
                            "Hiện chưa đủ cơ sở để đề xuất khoa khám. "
                            "Bạn có thể cung cấp thêm thông tin "
                            "về triệu chứng không?"
                        ),
                    },
                    finished=False,
                )

            self.awaiting_department_confirmation = True

            return self._build_result(
                ranked_diseases=ranked_diseases,
                next_question={
                    "symptom": None,
                    "question_type": "department_confirmation",
                    "question": (
                        f"Dựa trên các triệu chứng bạn cung cấp, "
                        f"khoa được đề xuất là {department}. "
                        "Đây là gợi ý định hướng khám, không phải "
                        "chẩn đoán xác định. Bạn có đồng ý với "
                        "đề xuất khoa khám này không?"
                    ),
                },
                finished=False,
            )
        if confirmation_value is False:

            self.awaiting_confirmation = False

            remaining_text = (
                result["remaining_text"]
            )

            # "chưa" nhưng không bổ sung gì
            if not remaining_text:
                if self.questions_asked >= self.max_questions:
                    return self._question_limit_result()

                ranked_diseases = self.rank_diseases()

                next_symptom = (
                    self.select_best_question_symptom(
                        ranked_diseases
                    )
                )

                if next_symptom is None:

                    confirmation_question = (
                        self.get_confirmation_question()
                    )

                    return self._build_result(
                        ranked_diseases=ranked_diseases,
                        next_question={
                            "symptom": None,
                            "question_type": "confirmation",
                            "question": (
                                confirmation_question
                            ),
                        },
                        finished=False,
                    )

                question = self.generate_question(
                    next_symptom
                )

                self.current_question = {
                    "symptom": next_symptom,
                    "question_type": "yes_no",
                    "question": question,
                }

                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    next_question=self.current_question,
                    finished=False,
                )


            return self._continue_from_free_form(
                remaining_text
            )

        # Không phải confirmation
        remaining_text = result["remaining_text"]
        if self.questions_asked >= self.max_questions:
            symptom_result = self.symptom_pipeline.process(
                remaining_text,
                update_state=False,
            )
            if not symptom_result.get("symptoms"):
                return self._question_limit_result()

        return self._continue_from_free_form(
            remaining_text
        )

    def _continue_from_free_form(
        self,
        text: str,
    ) -> dict:

        if not text.strip():

            return self._build_result(
                ranked_diseases=self.rank_diseases(),
                next_question=None,
                finished=False,
            )

        extracted_symptoms = self._record_free_form_symptoms(text)

        # Không hiểu user đang nói symptom gì
        if not extracted_symptoms:

            clarification_question = (
                self.get_clarification_question()
            )

            return self._build_result(
                ranked_diseases=self.rank_diseases(),
                next_question={
                    "symptom": None,
                    "question_type": "unclear_symptom",
                    "question": (
                        clarification_question
                    ),
                },
                finished=False,
            )

        # Rank lại sau khi bổ sung thông tin
        ranked_diseases = self.rank_diseases()

        detail_question = self._next_symptom_details_question()
        if detail_question is not None:
            return self._build_result(
                ranked_diseases=ranked_diseases,
                next_question=detail_question,
                finished=False,
            )

        # Nếu đủ thông tin → hỏi confirmation
        if self.should_stop(ranked_diseases):

            self.awaiting_confirmation = True
            self.current_question = None

            confirmation_question = (
                self.get_confirmation_question()
            )

            return self._build_result(
                ranked_diseases=ranked_diseases,
                next_question={
                    "symptom": None,
                    "question_type": "confirmation",
                    "question": confirmation_question,
                },
                finished=False,
            )

        # Chưa đủ → hỏi tiếp
        next_symptom = (
            self.select_best_question_symptom(
                ranked_diseases
            )
        )

        if next_symptom is None:

            confirmation_question = (
                self.get_confirmation_question()
            )

            return self._build_result(
                ranked_diseases=ranked_diseases,
                next_question={
                    "symptom": None,
                    "question_type": "confirmation",
                    "question": (
                        confirmation_question
                    ),
                },
                finished=False,
            )

        question = self.generate_question(
            next_symptom
        )

        self.current_question = {
            "symptom": next_symptom,
            "question_type": "yes_no",
            "question": question,
        }

        return self._build_result(
            ranked_diseases=ranked_diseases,
            next_question=self.current_question,
            finished=False,
        )

    def _is_end_intent(self, text: str) -> bool:
                    normalized = " ".join(text.lower().strip().split())

                    end_phrases = {
                        "kết thúc",
                        "hết rồi",
                        "tôi muốn dừng",
                        "mình muốn dừng",
                        "không hỏi nữa",
                        "dừng hội thoại",
                        "dừng lại",
                        "ko",
                        "Không",
                    }

                    return normalized in end_phrases



    def _handle_department_confirmation(
        self,
        text: str,
    ) -> dict:
        normalized = " ".join(
            text.lower().strip().split()
        ).rstrip("!.,")

        yes_phrases = {
            "có",
            "đồng ý",
            "tôi đồng ý",
            "đồng ý với đề xuất",
            "được",
            "ok",
            "okay",
            "vâng",
            "đúng rồi",
        }

        no_phrases = {
            "không",
            "không đồng ý",
            "tôi không đồng ý",
            "chưa đồng ý",
            "không muốn",
        }

        ranked_diseases = self.rank_diseases()

        if normalized in yes_phrases:
            self.awaiting_department_confirmation = False

            result = self._build_result(
                ranked_diseases=ranked_diseases,
                next_question=None,
                finished=True,
            )
            result["status"] = "completed"
            return result

        if normalized in no_phrases:
            self.awaiting_department_confirmation = False

            next_symptom = (
                self.select_best_question_symptom(
                    ranked_diseases
                )
            )

            if next_symptom is not None:
                question = self.generate_question(
                    next_symptom
                )

                self.current_question = {
                    "symptom": next_symptom,
                    "question_type": "yes_no",
                    "question": question,
                }

                return self._build_result(
                    ranked_diseases=ranked_diseases,
                    next_question=self.current_question,
                    finished=False,
                )

            return self._build_result(
                ranked_diseases=ranked_diseases,
                next_question={
                    "symptom": None,
                    "question_type": "unclear_symptom",
                    "question": (
                        "Bạn có thể cho biết thêm triệu chứng "
                        "hoặc lý do bạn chưa đồng ý với "
                        "khoa được đề xuất không?"
                    ),
                },
                finished=False,
            )

        return self._build_result(
            ranked_diseases=ranked_diseases,
            next_question={
                "symptom": None,
                "question_type": "department_confirmation",
                "question": (
                    "Bạn vui lòng cho biết có đồng ý với "
                    "khoa khám được đề xuất không? "
                    "Bạn cũng có thể bổ sung thông tin "
                    "nếu cần xem xét lại."
                ),
            },
            finished=False,
        )