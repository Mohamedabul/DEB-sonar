import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_fetching import DataFetching
from src.exception_handler import CustomException


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine and connection."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    return mock_engine


@patch("src.data_fetching.DataIngestion")
@patch("src.data_fetching.inspect")
@patch("pandas.read_sql")
def test_data_fetching_success(mock_read_sql, mock_inspect, MockDataIngestion, mock_engine):
    # Arrange
    mock_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    mock_read_sql.return_value = mock_df
    mock_inspect.return_value.get_table_names.return_value = ["users"]

    # Mock DataIngestion so it returns our mock_engine
    instance = MockDataIngestion.return_value
    instance.initiate_data_ingestion.return_value = mock_engine

    fetcher = DataFetching("users")

    # Act
    df = fetcher.data_fetching()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["id", "name"]


@patch("src.data_fetching.DataIngestion")
@patch("src.data_fetching.inspect")
@patch("pandas.read_sql", side_effect=Exception("DB error"))
def test_data_fetching_failure(mock_read_sql, mock_inspect, MockDataIngestion):
    # Arrange
    instance = MockDataIngestion.return_value
    instance.initiate_data_ingestion.return_value = MagicMock()

    fetcher = DataFetching("users")

    # Act & Assert
    with pytest.raises(CustomException):
        fetcher.data_fetching()
