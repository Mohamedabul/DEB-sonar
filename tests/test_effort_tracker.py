import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.effort_tracker import EffortTracking, EffortTrackingConfig


@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame similar to expected input structure."""
    return pd.DataFrame({
        'Project / Workstream Name': ['Proj1', 'Proj1'],
        'Deliverable': ['Del1', 'Del2'],
        'Variance': [1, -2],
        'EAC (Estimate at Completion)': [10, 20],
        'ETC (Estimate to Complete)': [2, 3],
        'Actual Effort Hours': [5, 15],
        'Actual End Date': ['2025-01-01', None],
        'Actual Start Date': ['2024-12-01', '2024-12-05'],
        'Estimated Effort (Hours)': [8, 12],
        'Sprint': ['Sprint 1', 'Sprint 2'],
        'Estimated\n Start Date': ['2024-11-01', '2024-11-10'],
        'Estimated\n End Date': ['2024-11-15', '2024-11-20'],
        'Capacity': [10, 20],
        'Team Member': ['Alice', 'Bob']
    })


@pytest.fixture
def patched_config(sample_dataframe):
    """Patch EffortTrackingConfig to return mock LLMs and sample dataframe."""
    with patch("src.effort_tracker.LLMInitializer") as MockLLMInit, \
         patch("src.effort_tracker.DataFetching") as MockDataFetching:

        # Mock LLMInitializer
        mock_llm = MagicMock()
        MockLLMInit.return_value.initialize_llm.return_value = mock_llm

        # Mock DataFetching
        mock_fetch = MagicMock()
        mock_fetch.data_fetching.return_value = sample_dataframe
        MockDataFetching.return_value = mock_fetch

        yield EffortTrackingConfig("fake_file.csv")


def test_effort_tracking_init(patched_config):
    """Test initialization of EffortTracking with mocked config."""
    tracker = EffortTracking("fake_file.csv")
    assert isinstance(tracker.config, EffortTrackingConfig)
    assert not tracker.config.df.empty


