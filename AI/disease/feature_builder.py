from typing import Dict

from AI.patient.state import PatientState


class DiseaseFeatureBuilder:

    def build(
        self,
        state: PatientState
    ) -> Dict[str, bool]:

        features = {}

        for code, symptom_state in state.symptoms.items():
            if symptom_state.value is True:
                features[code] = True

        return features