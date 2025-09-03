import pytest
from langchain.prompts import ChatPromptTemplate
from src.prompts import Router, EffortAnalysisPrompt, EffortAnalysisPrompt2


def test_router_is_chatprompttemplate():
    """Ensure Router is a ChatPromptTemplate instance."""
    assert isinstance(Router, ChatPromptTemplate)


def test_router_formatting_user_request():
    """Check that Router formats a user request correctly."""
    formatted = Router.format(user_request="Show me sprint utilization for sprint 2")
    assert "User Request: Show me sprint utilization for sprint 2" in formatted
    assert "system" in formatted or "Role: Manager" in formatted


def test_effort_analysis_prompt_formatting():
    """Check EffortAnalysisPrompt generates properly formatted output."""
    formatted = EffortAnalysisPrompt.format(
        question="What is Alice's progress?",
        sprint_details="Alice has 5 tasks pending in Sprint 1."
    )
    assert "Question: What is Alice's progress?" in formatted
    assert "Sprint Details: Alice has 5 tasks pending in Sprint 1." in formatted


def test_effort_analysis_prompt2_formatting():
    """Check EffortAnalysisPrompt2 works with Burn Down Report input."""
    formatted = EffortAnalysisPrompt2.format(
        report="Sprint 1 burn down shows stalled progress.",
        question="Why did sprint 1 stall?",
        sprint_details="Pavan had blockers in Task #12."
    )
    assert "Burn Down Report: Sprint 1 burn down shows stalled progress." in formatted
    assert "Question: Why did sprint 1 stall?" in formatted
    assert "Sprint Details: Pavan had blockers in Task #12." in formatted


@pytest.mark.parametrize("template,fields", [
    (Router, ["user_request"]),
    (EffortAnalysisPrompt, ["question", "sprint_details"]),
    (EffortAnalysisPrompt2, ["report", "question", "sprint_details"]),
])
def test_input_variables(template, fields):
    """Ensure each prompt has expected input variables."""
    assert sorted(template.input_variables) == sorted(fields)
