"""Reject same-split data from a different admission receipt before model loading."""
from types import SimpleNamespace
from eval.action4_staged_eval import validate_reused_data
from data.action4_paths import PROJECT_ROOT as R

def main():
    receipt = "outputs/required/DATA_VALIDATED_ALL_SPLITS.json"
    data = SimpleNamespace(role="B0", records=[{"split": "eval"}],
                           _validation_receipt=(R / receipt).resolve())
    validate_reused_data(data, receipt, "eval")
    cases = [
        SimpleNamespace(role="B0", records=data.records),
        SimpleNamespace(role="B0", records=data.records, _validation_receipt=R / "wrong.json"),
        SimpleNamespace(role="A0", records=data.records, _validation_receipt=data._validation_receipt),
        SimpleNamespace(role="B0", records=[{"split": "test"}], _validation_receipt=data._validation_receipt),
        SimpleNamespace(role="B0", records=[], _validation_receipt=data._validation_receipt),
    ]
    for candidate in cases:
        try:
            validate_reused_data(candidate, receipt, "eval")
        except ValueError:
            continue
        raise AssertionError("Invalid reused evaluation input accepted")
    print("PASS: matching receipt accepted; five wrong-identity/role/split/empty cases rejected")

if __name__ == "__main__":
    main()
