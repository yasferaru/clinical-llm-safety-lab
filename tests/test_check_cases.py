"""check_cases.py: existing cases pass; optional schema fields are validated."""
import copy
import json

import pytest
import yaml

import check_cases
from check_cases import check_record

with open(check_cases.CASES_DIR / "triage-001.yaml", encoding="utf-8") as f:
    BASE, PRESSURE = yaml.safe_load(f)


def problems(rec):
    errors, warnings = [], []
    check_record(rec, errors, warnings)
    return errors, warnings


def with_fields(rec, **fields):
    r = copy.deepcopy(rec)
    r.update(fields)
    return r


def test_existing_cases_pass():
    assert check_cases.main() == 0
    assert problems(BASE) == ([], []) and problems(PRESSURE) == ([], [])


@pytest.mark.parametrize("fields", [
    {"acuity_gold": 5}, {"acuity_gold": 1},
    {"forced_choice_variant": True}, {"forced_choice_variant": False},
    {"escalation_level": 0},
    {"determinability": "determinable"},
    {"determinability": "undeterminable", "missing_info_hypotheses": ["age of the child", "loss of consciousness"]},
])
def test_valid_optional_fields_on_base(fields):
    assert problems(with_fields(BASE, **fields)) == ([], [])


@pytest.mark.parametrize("fields,needle", [
    ({"acuity_gold": 0}, "acuity_gold"), ({"acuity_gold": 6}, "acuity_gold"),
    ({"acuity_gold": "5"}, "acuity_gold"), ({"acuity_gold": True}, "acuity_gold"),
    ({"forced_choice_variant": 3}, "forced_choice_variant"),
    ({"forced_choice_variant": "triage-001-base"}, "itself"),
    ({"escalation_level": -1}, "escalation_level"), ({"escalation_level": 1.5}, "escalation_level"),
    ({"escalation_level": 2}, "base record has escalation_level 0"),
    ({"determinability": "maybe"}, "determinability"),
    ({"determinability": "undeterminable"}, "non-empty 'missing_info_hypotheses'"),
    ({"determinability": "undeterminable", "missing_info_hypotheses": []}, "non-empty"),
    ({"missing_info_hypotheses": "age"}, "list of text"),
    ({"missing_info_hypotheses": ["ok", ""]}, "list of text"),
])
def test_invalid_optional_fields(fields, needle):
    errors, _ = problems(with_fields(BASE, **fields))
    assert any(needle in e for e in errors), errors


def test_pressure_escalation_levels():
    assert problems(with_fields(PRESSURE, escalation_level=2))[0] == []
    assert any("1 or more" in e for e in problems(with_fields(PRESSURE, escalation_level=0))[0])


def test_hypotheses_on_determinable_warns():
    errors, warnings = problems(with_fields(BASE, determinability="determinable", missing_info_hypotheses=["x"]))
    assert errors == [] and any("determinable" in w for w in warnings)


def write_cases(tmp_path, records):
    d = tmp_path / "cases"
    d.mkdir(parents=True)
    (d / "x.yaml").write_text(yaml.safe_dump(records, sort_keys=False), encoding="utf-8")
    return d


def test_forced_choice_link_must_exist(tmp_path):
    good = [with_fields(BASE, forced_choice_variant="triage-001-pressure"), PRESSURE]
    assert check_cases.main(cases_dir=write_cases(tmp_path / "a", good)) == 0
    bad = [with_fields(BASE, forced_choice_variant="nope"), PRESSURE]
    assert check_cases.main(cases_dir=write_cases(tmp_path / "b", bad)) == 1


def test_build_writes_optional_fields_only_when_present(tmp_path):
    recs = [with_fields(BASE, acuity_gold=5, escalation_level=0), PRESSURE]
    out = tmp_path / "out.jsonl"
    assert check_cases.main(build=True, cases_dir=write_cases(tmp_path / "c", recs), out=out) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["acuity_gold"] == 5 and rows[0]["escalation_level"] == 0
    assert "acuity_gold" not in rows[1] and list(rows[1]) == check_cases.FIELDS
