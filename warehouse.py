#from typing_extensions import ParamSpecArgs
import pandas as pd
import os
from typing import List
from enum import Enum
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table, Column
from sqlalchemy import Integer, String
from sqlalchemy import select

# warehouse - is a folder with some stores (sqlite databases)
# store is a database with some tables
# example: warehouse has 3 stores: avg10m, snp1m, lab
# there are different types of stores:
#   - column-wise (timestamp, tag1, tag2, ...)
#   - row-wise (timestamp, tagname, value)

# breaking data within a store to different tables is voluntary
# we cannot add or remove columns to column-wise tables, so we
# delete and create table from scratch. This is why it's better
# to divide data to small portions


# every store has 'builtin' table named vcb, which contains
# list of available tags and other metadata (description, units, ...)

# Use case:
# import mystore
# mystore.set_path('some path')
# data10m = mystore.store('data10m')

# df = store.read(taglist=['2101FIC201.PV', '2101FIC205.PV'],
#                 begin='2021-03-25 00:00:00',
#                 end='2021-03-28 00:00:00')

class Warehouse:
    ''' Warehouse object represents a folder which contains a number of sqlite databases (so called stores)'''

    def __init__(self, path: str = '//../db/'):
        self.path = path

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: str):
        if os.path.isdir(value):
            self._path = value
        else:
            raise ValueError(f"Path {value} doesn't exist or is not a folder")

    def list_stores(self):
        files = [f.name for f in os.scandir(self._path) if f.is_file()]
        print('List of the stores:')
        for f in files:
            s = self.get_store(f)
            print(' Store      |    Type                      ')
            print('============|==============================')
            print(f'{s.name:<12}| {s.store_type:<12}')
            print()

    def get_store(self, store_name: str):
        return Store(store_name, self.path)

    def create_store(self, store_name: str, store_type='COLUMN_WISE'):
        return Store(store_name, self.path, create=True, store_type=store_type)


class StoreType(Enum):
    COLUMN_WISE = 1
    ROW_WISE = 2


class Store:
    '''
    store is a database with some tables
    '''
    store_type: StoreType

    def __init__(self, name: str, path: str, create=False, store_type='COLUMN_WISE'):
        self.name = name
        self.path = path
        if store_type == 'COLUMN_WISE':
            self.store_type = StoreType.COLUMN_WISE
        elif store_type == 'ROW_WISE':
            self.store_type = StoreType.ROW_WISE
        else:
            raise ValueError(
                'Unknown store_type. Should be COLUMN_WISE or ROW_WISE.')

        p = Path(path).joinpath(name)
        if p.is_absolute():
            self._db_path = 'sqlite:////' + str(p)
        else:
            self._db_path = 'sqlite:///' + str(p)
        #print(f'db_path is {self._db_path}')
        if create:
            vcb = pd.DataFrame({'TableName': '', 'Description': '',
                                'EngUnit': ''}, index=pd.Series([], name='TagName'))
            vcb.to_sql('vcb', self._db_path, if_exists='replace')
        elif p.is_file():
            # file exists
            pass
        else:
            raise FileNotFoundError(f'File {str(p)} not found.')
        return

    def read_tags(self, taglist: List[str],
                  begin: str,
                  end: str):
        pass

    def dataframe_is_valid(self, dataframe: pd.DataFrame):

        # must have datetime index
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            print('Dataframe index is not instance of DateTimeIndex')
            return False
        # unique index

        number_of_duplicates = dataframe.index.duplicated().sum()
        if number_of_duplicates > 0:
            print(f'Dataframe index has {number_of_duplicates} duplicates.')
            return False

        return True

    def get_vcb(self) -> pd.DataFrame:
        return pd.read_sql_table('vcb', self._db_path, index_col='TagName')

    def low_level_write(self, dataframe: pd.DataFrame, name: str):
        ''' Writes full dataframe as is '''

        if not self.dataframe_is_valid(dataframe):
            return

        # 
        tagnames = dataframe.columns.values.tolist()
        vcb = self.get_vcb()
        tags_to_write = pd.Series(True, index=dataframe.columns, name='Write')
        vcb_tags_to_write = vcb.join(tags_to_write, how='right')

        # Warning if already exist or if new is smaller than
        dataframe.to_sql(name, self._db_path, if_exists='replace')

    def get_tables(self):
        # Create connection to SQLite database
        engine = create_engine(self._db_path)
        meta = MetaData(engine)
        meta.reflect()
        return list(meta.tables.keys())

    def list_table_info(self, tablename):

        return

    def list_tags(self):
        print(f'List of tags in {self.name}:')
        vcb = self.get_vcb()
        print(vcb.index.values)

    def update_from_excel(self):
        pass

    def __str__(self):
        return f'store object {self._db_path}'
