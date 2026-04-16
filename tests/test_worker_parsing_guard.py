from __future__ import annotations

from common.parsing import parse_model_output


def test_parse_model_output_does_not_infer_confidence_from_unrelated_numbers() -> None:
    output = (
        "The claim is false. The treaty was ratified in 2010 and limits deployed warheads "
        "to 1,550, but the statement overstates what it does."
    )

    prediction = parse_model_output(output)

    assert prediction.label == "false"
    assert prediction.confidence == 0.5
