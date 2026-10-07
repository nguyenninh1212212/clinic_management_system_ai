from AI.patient import PatientStateService


service = PatientStateService()


# Bệnh nhân bị đau đầu 3 ngày
service.update_symptom(
    code="headache",
    value=True,
    duration="3 ngày",
)


# Bệnh nhân bị chóng mặt
service.update_symptom(
    code="dizziness",
    value=True,
)


# Bệnh nhân không bị sốt
service.update_symptom(
    code="fever",
    value=False,
)


state = service.get_state()


for code, symptom in state.symptoms.items():

    print(
        code,
        "->",
        {
            "value": symptom.value,
            "duration": symptom.duration,
            "severity": symptom.severity,
            "location": symptom.location,
        },
    )