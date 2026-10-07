from AI.question.answer_parser_free_form import (
    FreeFormAnswerParser,
)


def test_duration_days():

    parser = FreeFormAnswerParser()

    result = parser.parse(
        "Tôi bị nôn 3 ngày rồi"
    )

    print("\nRESULT:", result)

    assert result["duration"] == "3 ngày"


def test_duration_from_morning():

    parser = FreeFormAnswerParser()

    result = parser.parse(
        "Tôi nôn từ sáng"
    )

    print("\nRESULT:", result)

    assert result["duration"] == "từ sáng"


def test_duration_from_yesterday():

    parser = FreeFormAnswerParser()

    result = parser.parse(
        "Tôi đau bụng từ hôm qua"
    )

    print("\nRESULT:", result)

    assert result["duration"] == "từ hôm qua"


def test_severity():

    parser = FreeFormAnswerParser()

    result = parser.parse(
        "Tôi đau đầu khá nặng"
    )

    print("\nRESULT:", result)

    assert result["severity"] == "severe"


def test_duration_and_severity():

    parser = FreeFormAnswerParser()

    result = parser.parse(
        "Tôi đau đầu 3 ngày rồi và đau rất nặng"
    )

    print("\nRESULT:", result)

    assert result["duration"] == "3 ngày"
    assert result["severity"] == "severe"