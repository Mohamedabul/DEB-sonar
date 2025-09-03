import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sqlalchemy.engine import Engine
from src.data_ingestion import DataIngestion
from src.exception_handler import CustomException


@patch("src.data_ingestion.pd.ExcelFile")
@patch("src.data_ingestion.pd.read_excel")
@patch("src.data_ingestion.pd.DataFrame.to_sql")   # ✅ mock to_sql
@patch("src.data_ingestion.create_engine")
def test_initiate_data_ingestion_success(mock_create_engine, mock_to_sql, mock_read_excel, mock_excel_file):
    # Arrange
    fake_engine = MagicMock(spec=Engine)
    mock_create_engine.return_value = fake_engine

    # Mock ExcelFile
    mock_excel_file.return_value.sheet_names = ["Sheet1"]

    # Mock read_excel to return a sample dataframe
    mock_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    mock_read_excel.return_value = mock_df

    ingestion = DataIngestion("test_table")

    # Act
    engine = ingestion.initiate_data_ingestion()

    # Assert
    assert engine == fake_engine
    mock_read_excel.assert_called_once()
    mock_to_sql.assert_called_once_with(
        "test_table", con=fake_engine, index=False, if_exists="replace", method="multi"
    )


@patch("src.data_ingestion.pd.ExcelFile", side_effect=FileNotFoundError("File not found"))
def test_initiate_data_ingestion_file_not_found(mock_excel_file):
    ingestion = DataIngestion("missing_table")
    with pytest.raises(CustomException) as exc_info:
        ingestion.initiate_data_ingestion()
    assert "File not found" in str(exc_info.value)
