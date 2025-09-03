import sys
import builtins
import pytest
from unittest.mock import MagicMock, patch
import importlib

# -------------------------------------------------------------------
# Fake session state to mimic Streamlit's attribute + dict access
# -------------------------------------------------------------------
class FakeSessionState(dict):
    """Dict-like object that also supports attribute-style access like Streamlit's session_state."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value


# -------------------------------------------------------------------
# Fixture: Patch Streamlit
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    mock_st = MagicMock()
    sys.modules["streamlit"] = mock_st

    # Default UI stubs
    mock_st.file_uploader.return_value = None
    mock_st.chat_input.return_value = None
    mock_st.selectbox.return_value = ""
    mock_st.session_state = FakeSessionState(
        uploaded_file=None,
        selected_project=None,
        result=None,
        show_chat=False,
        messages=[]
    )

    return mock_st


# -------------------------------------------------------------------
# Fixture: Patch WeasyPrint (avoid loading native libs)
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_weasyprint(monkeypatch):
    fake_weasyprint = MagicMock()
    sys.modules["weasyprint"] = fake_weasyprint
    return fake_weasyprint


# -------------------------------------------------------------------
# TEST 1: Ensure app imports successfully
# -------------------------------------------------------------------
def test_app_imports_without_errors(patch_streamlit, patch_weasyprint):
    with patch("src.effort_tracker.EffortTracking") as MockEffortTracking, \
         patch("src.utils.clear_analysis_outputs", return_value=None):
        import app
        importlib.reload(app)  # reload to apply mocks
        assert MockEffortTracking is not None


# -------------------------------------------------------------------
# TEST 2: Chat flow should call EffortTracking
# -------------------------------------------------------------------
def test_chat_flow_triggers_effort_analysis(patch_streamlit, patch_weasyprint):
    with patch("src.effort_tracker.EffortTracking") as MockEffortTracking, \
         patch("src.utils.clear_analysis_outputs", return_value=None):

        mock_tracker = MockEffortTracking.return_value
        mock_tracker.effort_analysis.return_value = "Analysis Complete"

        # Simulate user interacting with chat
        patch_streamlit.session_state = FakeSessionState(
            uploaded_file=None,
            selected_project="Orchestration",
            result=None,
            show_chat=True,
            messages=[],
        )
        patch_streamlit.chat_input.return_value = "Test user question"

        import app
        importlib.reload(app)

        mock_tracker.effort_analysis.assert_called_once_with("Test user question")


# -------------------------------------------------------------------
# TEST 3: File upload branch works
# -------------------------------------------------------------------
def test_file_upload_creates_file(tmp_path, patch_streamlit, patch_weasyprint):
    fake_file = MagicMock()
    fake_file.name = "dummy.xlsx"
    fake_file.getbuffer.return_value = b"dummy content"

    # Patch file_uploader to return our fake file
    patch_streamlit.file_uploader.return_value = fake_file

    # Patch open to write into tmp_path instead of project root
    with patch("builtins.open", builtins.open):
        with patch("os.path.join", side_effect=lambda *a: str(tmp_path / a[-1])):
            with patch("src.effort_tracker.EffortTracking"):
                with patch("src.utils.clear_analysis_outputs", return_value=None):
                    import app
                    importlib.reload(app)

    uploaded_path = tmp_path / "dummy.xlsx"
    assert uploaded_path.exists()
