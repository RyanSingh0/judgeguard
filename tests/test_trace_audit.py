import json

import pytest

from judgeguard.agent.audit import audit_trace, trace_examples


@pytest.mark.parametrize("name", ["Valid execution", "Redundant but valid call"])
def test_valid_calls_and_redundancy_pass(name):
    assert audit_trace(trace_examples()[name])["status"] == "passed"


@pytest.mark.parametrize(
    "name",
    [
        "Wrong argument, correct final answer",
        "Fabricated tool result",
        "Undeclared tool",
        "Wrong final answer",
    ],
)
def test_constructed_defects_fail(name):
    assert audit_trace(trace_examples()[name])["status"] == "failed"


def test_correct_answer_does_not_hide_bad_execution():
    result = audit_trace(trace_examples()["Wrong argument, correct final answer"])
    assert result["answer_matches_expected"] is True
    assert result["step_failures"] == 1
    assert result["steps"][2]["replayed"] == "119700"


def test_without_oracle_never_claims_task_success():
    trace = trace_examples()["Valid execution"]
    del trace["expected_final_answer"]
    result = audit_trace(trace)
    assert result["status"] == "consistent"
    assert result["answer_matches_expected"] is None


@pytest.mark.parametrize("expr", ["1/0", "__import__('os').getcwd()", "2 ** 999", "(2**8)**8**8"])
def test_invalid_and_unsafe_expressions_are_contained(expr):
    trace = trace_examples()["Valid execution"]
    trace["steps"][2]["args"]["expr"] = expr
    assert audit_trace(trace)["steps"][2]["status"] == "invalid_call"


def test_input_limits_and_no_nested_arguments():
    trace = trace_examples()["Valid execution"]
    trace["steps"][0]["args"] = {"key": {"nested": "value"}}
    with pytest.raises(ValueError):
        audit_trace(trace)
    with pytest.raises(ValueError):
        audit_trace(" " * 100001)


def test_serialized_trace_uses_same_checks():
    trace = trace_examples()["Valid execution"]
    assert audit_trace(json.dumps(trace)) == audit_trace(trace)
