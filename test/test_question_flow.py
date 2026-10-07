from AI.question.answer_parser import AnswerParser


def test_simple_yes():
    parser = AnswerParser()

    result = parser.parse("Có")

    print(f'\nparse("Có") = {result}')

    assert result is True


def test_simple_no():
    parser = AnswerParser()

    result = parser.parse("Không")

    print(f'\nparse("Không") = {result}')

    assert result is False


def test_yes_with_context():
    parser = AnswerParser()

    result = parser.parse_with_context(
        "Có",
        "vomiting",
    )

    print(f'\nparse_with_context("Có", "vomiting") = {result}')

    assert result is True


def test_no_with_context():
    parser = AnswerParser()

    result = parser.parse_with_context(
        "Không",
        "vomiting",
    )

    print(f'\nparse_with_context("Không", "vomiting") = {result}')

    assert result is False


def test_negation():
    parser = AnswerParser()

    result = parser.is_negated(
        "Tôi không bị sốt"
    )

    print(f'\nis_negated("Tôi không bị sốt") = {result}')

    assert result is True


def test_free_form_answer():
    parser = AnswerParser()

    result = parser.parse(
        "Có, tôi nôn 2 lần từ sáng"
    )

    print(
        f'\nparse("Có, tôi nôn 2 lần từ sáng") = {result}'
    )

    assert result is None
