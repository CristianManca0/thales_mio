"""Preprocessor class for data preprocessing pipelines."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .utils import convert_to_numeric_raw, load_encoders_raw


class RawPreprocessor:
    """ Preprocessor class to handle training and testing preprocessing pipelines."""

    def train(
        self,
        X: pd.DataFrame,
        data_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Call the train preprocessing pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Training clean DataFrame.
        data_path : Path | str | None
            Path to load/save the preprocessed data. If specified, the preprocessed data
            will be saved to this path after preprocessing. If a file exists at this path,
            it will be loaded instead of preprocessing the data again.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if data_path is not None and Path(data_path).exists():
            with Path(data_path).open("r") as f:
                return pd.read_csv(f, sep=";")

        (Path(__file__).parent / "models_preprocessing").mkdir(exist_ok=True)

        df = X.copy()
        df, cat_tags = convert_to_numeric_raw(df)

        encoder = load_encoders_raw()
        string_cols = [col for col, tag in cat_tags if tag == "string"]
        if string_cols:
            df[string_cols] = encoder.fit_transform(df[string_cols].astype(str))
            joblib.dump(encoder, Path(__file__).parent / "models_preprocessing/ordinal_encoder_raw.pkl")

        scaler_path = Path(__file__).parent / "models_preprocessing/scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            scaler = RobustScaler()
        df_final = scaler.fit_transform(df)
        df_final = pd.DataFrame(df_final, columns=df.columns, index=df.index)

        if not scaler_path.exists():
            joblib.dump(
                scaler,
                Path(__file__).parent / "models_preprocessing/scaler.pkl",
            )

        if data_path is not None:
            Path(data_path.parent).mkdir(exist_ok=True)
            df_final.to_csv(data_path, sep=";", index=False)

        return df_final


    def test(
        self,
        X: pd.DataFrame,
        data_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Call the test preprocessing pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Test clean DataFrame.
        data_path : Path | str | None
            Path to load/save the preprocessed data. If specified, the preprocessed data
            will be saved to this path after preprocessing. If a file exists at this path,
            it will be loaded instead of preprocessing the data again.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if data_path is not None and Path(data_path).exists():
            with Path(data_path).open("r") as f:
                return pd.read_csv(f, sep=";")

        df = X.copy()
        df, cat_tags = convert_to_numeric_raw(df)

        encoder = load_encoders_raw()
        string_cols = [col for col, tag in cat_tags if tag == "string"]
        if string_cols:
            df[string_cols] = encoder.transform(df[string_cols].astype(str))

        scaler_path = Path(__file__).parent / "models_preprocessing/scaler.pkl"
        if not scaler_path.exists():
            raise SystemError(
                "Missing scaler model. Please run training first."
            )

        scaler: RobustScaler = joblib.load(scaler_path)
        df_final = scaler.transform(df)
        df_final = pd.DataFrame(df_final, columns=df.columns, index=df.index)

        if data_path is not None:
            df_final.to_csv(data_path, sep=";", index=False)

        return df_final
