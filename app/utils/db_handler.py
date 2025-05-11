from utils.config_loader import ConfigLoader
from sqlalchemy import create_engine
import pandas as pd


def get_engine():
    cfg = ConfigLoader.get_mysql_config()
    uri = f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}/{cfg['database']}"
    return create_engine(uri)

def save_dataframe(df: pd.DataFrame, table_name: str, if_exists="replace"):
    engine = get_engine()
    df.to_sql(name=table_name, con=engine, index=False, if_exists=if_exists)

def load_dataframe(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
