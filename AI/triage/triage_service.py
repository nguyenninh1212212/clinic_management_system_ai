from collections import defaultdict

from AI.department.knowledge_base import (
    DiseaseDepartmentKnowledgeBase,
)


class DepartmentTriageService:
    def __init__(
        self,
        knowledge_base: DiseaseDepartmentKnowledgeBase,
    ):
        self.knowledge_base = knowledge_base

    def suggest_department(
        self,
        ranked_diseases: list,
    ) -> dict:
        if not ranked_diseases:
            return {
                "suggested_department": None,
                "score": 0.0,
                "departments": [],
                "status": "insufficient_evidence",
            }

        department_diseases = defaultdict(list)

        for item in ranked_diseases:
            disease = item.get("disease")
            disease_score = item.get("score", 0.0)

            if not disease:
                continue

            department = self.knowledge_base.get_department(
                disease
            )

            if not department:
                continue

            department_diseases[department].append({
                "disease": disease,
                "score": disease_score,
            })

        if not department_diseases:
            return {
                "suggested_department": None,
                "score": 0.0,
                "departments": [],
                "status": "insufficient_evidence",
            }

        department_scores = []

        for department, diseases in department_diseases.items():
            scores = [
                item["score"]
                for item in diseases
            ]

            # Lấy score cao nhất làm đại diện chính
            # và cộng thêm một phần hỗ trợ từ các candidates khác.
            top_score = max(scores)

            top_index = scores.index(top_score)
            supporting_score = sum(
                score
                for index, score in enumerate(scores)
                if index != top_index
            )

            department_score = (
                top_score
                + 0.25 * supporting_score
            )

            department_score = min(
                department_score,
                1.0,
            )

            department_scores.append({
                "department": department,
                "score": round(
                    department_score,
                    4,
                ),
                "diseases": [
                    item["disease"]
                    for item in diseases
                ],
            })

        department_scores.sort(
            key=lambda item: (
                -item["score"],
                item["department"],
            ),
        )

        best_department = department_scores[0]
        tied_departments = [
            item
            for item in department_scores
            if item["score"] == best_department["score"]
        ]

        if len(tied_departments) > 1:
            return {
                "suggested_department": None,
                "score": best_department["score"],
                "departments": department_scores,
                "status": "ambiguous",
            }

        return {
            "suggested_department": (
                best_department["department"]
            ),
            "score": best_department["score"],
            "departments": department_scores,
            "status": "suggested",
        }