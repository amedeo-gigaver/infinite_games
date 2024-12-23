from typing import List

import pandas as pd
from pydantic import BaseModel


def pydantic_models_to_dataframe(models: List[BaseModel]) -> pd.DataFrame:
    if not models:
        return pd.DataFrame()

    # Extract the field schema from the first model
    field_info = models[0].model_fields

    records = [m.model_dump() for m in models]

    df = pd.DataFrame(records)

    # Build a mapping from Python type to pandas/numpy dtype
    # This is a basic mapping and can be expanded as needed.
    type_mapping = {
        int: "Int64",  # pandas nullable integer type
        float: "float64",
        str: "string",  # pandas nullable string type
        bool: "boolean",  # pandas nullable boolean type
        # For datetime fields, pandas usually autodetects dtype='datetime64[ns]'
        # You can add custom handling if needed.
    }

    # Try to set the appropriate column dtypes based on annotations
    for field_name, field_def in field_info.items():
        if field_name in df.columns:
            # field_def.annotation holds the Python type
            py_type = field_def.annotation
            # If the annotation is something like Optional[int], extract the main type
            # In Pydantic v2, annotation might be a typing.Union or similar. Handle gracefully:
            origin = getattr(py_type, "__origin__", None)
            if origin is not None and origin is not str and hasattr(py_type, "__args__"):
                # Take the first arg as a guess (e.g., Union[int, None])
                py_type = py_type.__args__[0]

            # Map the Python type to a pandas dtype if possible
            mapped_dtype = type_mapping.get(py_type, None)
            if mapped_dtype is not None:
                # Convert the column
                df[field_name] = df[field_name].astype(mapped_dtype, errors="ignore")

    return df
