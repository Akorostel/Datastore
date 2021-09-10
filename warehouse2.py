#from typing_extensions import ParamSpecArgs
from warnings import WarningMessage
import numpy as np
import pandas as pd
import os
from typing import Dict, List
from enum import Enum
from pathlib import Path
from fastparquet import ParquetFile
import warnings
import uuid
import openpyxl
import logging

# Warehouse is a folder with parquet files. Each parquet file holds dataframe with datetime index and several columns (tags)
# Example: Folder with 3 warehouses: avg10m, snp1m, lab

# When you add data to the store, they are written 10 tags per file by default. 
# This can be reassigned manually when calling write_tags, or warehouse can be fully rebuilt later

# We cannot add or remove columns to file, so we
# delete and create file from scratch. This is why it's better
# to divide data to small portions

# Every folder has 'builtin' parquet file named vcb, which contains
# list of available tags, tag <-> file mapping, and (under question) other metadata (description, units, ...)

# The following is under question:
# There are different types of files (or warehouses?):
#   - column-wise (timestamp, tag1, tag2, ...)
#   - row-wise (timestamp, tagname, value)

# Use case:
# import warehouse as wh
# store = wh.create_store('some path')
# store = wh.connect_store('some_path')

# store.list_tags()
# store.get_vcb()
# store.get_info()

# df = store.read(taglist=['2101FIC201.PV', '2101FIC205.PV'],
#                 begin='2021-03-25 00:00:00',
#                 end='2021-03-28 00:00:00')

# store.write(df)
# store.write(df, vcb)

# store.rebuild(vcb)

# store.update_from_excel('some_folder_path')

class Store:
    ''' Store object represents a folder which contains a number of parquet files'''

    def __init__(self, path: str = '//wh', tags_per_file=10):
        self.path = path
        if tags_per_file < 1:
            tags_per_file = 1
            warnings.warn('tags_per_file set to 1.')
        elif tags_per_file > 1000:
            tags_per_file = 1000
            warnings.warn('tags_per_file set to 1000.')

        self.tags_per_file = tags_per_file

        p = Path(path).joinpath('vcb.parquet')
        if p.is_file(): # file exists
            self._vcb = pd.read_parquet(p)
            print('Succesfully connected to existing store.')
        else:
            #TODO: check if other files already present in the folder?
            self._vcb = pd.DataFrame({'Filename': [], 'Description': [], 'EngUnit': []}, 
                    index=pd.Series([], name='TagName', dtype='object'))
            self._vcb.to_parquet(p, index=True)
            print('Created new store.')


    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: str):
        if os.path.isdir(value):
            self._path = value
        else:
            raise ValueError(f"Path {value} doesn't exist or is not a folder")

    def file_info(self):
        '''Get list of files used in the store with short information.
        Returns list of dicts with keys: filename, cols, rows, begin, end'''
        files = [f for f in os.scandir(self._path) if f.is_file()]
        info = []
        for f in files:  
            try:
                pf = ParquetFile(f.path)
                cols = len(pf.info['columns'])
                rows = pf.info['rows']
                try:
                    begin = str(pf.statistics['min']['Timestamp'][0])[:16]
                    end = str(pf.statistics['max']['Timestamp'][0])[:16]
                except:
                    begin, end = '', ''
            except:
                cols, rows, begin, end = 'None', 'None', 'None', 'None'
            info.append({'filename': f.name, 'cols': cols, 'rows': rows, 'begin': begin, 'end': end})
        return info

    def list_files(self):
        '''Get dataframe with short information about files used in the store.'''
        return pd.DataFrame(self.file_info())

    def dataframe_is_valid(self, dataframe: pd.DataFrame):
        '''Check if dataframe can be written to the store'''
        # must have datetime index
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            print('Dataframe index is not instance of DateTimeIndex')
            return False

        # index must be unique
        number_of_duplicates = dataframe.index.duplicated().sum()
        if number_of_duplicates > 0:
            print(f'Dataframe index has {number_of_duplicates} duplicates.')
            return False

        return True

    def rebuild_vcb(self):
        '''Rebuild vcb dataframe with tag <-> file mapping, which is
        built from real files. '''

        files = [f for f in os.scandir(self._path) if f.is_file() and 
                                                      f.name != 'vcb.parquet' and 
                                                      'metadata' not in f.name]
        tagname_list = []
        filename_list = []
        description_list = []
        engunit_list = []

        for f in files:
            try:
                pf = ParquetFile(f.path)
            except:
                warnings.warn(f'{f.name} is not parquet file.')
                continue

            columns = pf.info['columns']
            if 'Timestamp' in columns:
                columns.remove('Timestamp')
            else:
                warnings.warn(f'{f.name} doesnt contain Timestamp column')
                continue
            
            for c in columns:
                tagname_list.append(c)
                filename_list.append(f.name)
                try:
                    description_list.append(self._vcb.loc[c, 'Description'])
                except KeyError:
                    description_list.append('')
                try:
                    engunit_list.append(self._vcb.loc[c, 'EngUnit'])
                except KeyError:
                    engunit_list.append('')

        self._vcb = pd.DataFrame({'Filename': filename_list, 
                                   'Description': description_list,
                                   'EngUnit': engunit_list}, 
                                    index=pd.Series(tagname_list, name='TagName'))
        self._vcb.to_parquet(Path(self._path).joinpath('vcb.parquet'))

        return self._vcb

    def update_file(self, dataframe: pd.DataFrame, filename: str):
        # TODO: check if time ranges of existing and new dataframes intersect, and generate warning 
        # TODO: check if samplerate is different, and generate warning

        p = Path(self.path).joinpath(filename)
        file_content = pd.read_parquet(p)
        updated_frame = pd.concat([file_content, dataframe])
        n_duplicated = updated_frame.index.duplicated().sum() # default keep='First', so old values are preserved
        if n_duplicated:
            logging.warning(f'While updating file {filename}, {n_duplicated} duplicated records were found.')
        updated_frame = updated_frame.loc[~updated_frame.index.duplicated()]
        updated_frame.sort_index(inplace=True)
        updated_frame.to_parquet(p)

    def write(self, dataframe: pd.DataFrame, new_vcb: pd.DataFrame = None):
        '''Write dataframe tags to the store.
        Tag<->file mapping will be done automatically, but can also
        be provided in new_vcb dataframe (not supported yet)'''
        #TODO: support for new_vcb

        # tags in dataframe:
        #   some already exist in the warehouse
        #       if file location is specified for existing tag and does not equal to its real location, then generate warning and exit
        #       otherwise save the tag to existing file location
        #   some are new
        #       if file location is specified, use it
        #       if not specified - generate new file for them
        #           or try to place new tags in existing files which have free space (this will be implemented later)
        
        # 1. Generate tag->file mapping: tag, file, existing
        tags_to_write = pd.Series(True, index=dataframe.columns, name='Write')
        vcb_tags_to_write = self._vcb.join(tags_to_write, how='right').sort_values(by='Filename')
        vcb_tags_to_write['Existing'] = ~vcb_tags_to_write['Filename'].isna()

        existing_files = vcb_tags_to_write[vcb_tags_to_write['Existing']]['Filename'].unique().tolist()
        unmapped_tags = vcb_tags_to_write[~vcb_tags_to_write['Existing']].index.tolist()
        # 2. For each existing file,
        #       Call update_file(file, dataframe)
        for f in existing_files:
            tags = vcb_tags_to_write[vcb_tags_to_write['Filename']==f].index.tolist()
            self.update_file(dataframe[tags], f)
            logging.info(f'Tags {tags} written to existing file {f}')
        # 3. For each non-existing file,
        #       Call dataframe.to_parquet()
        tags_per_file = 10
        tag_groups = [unmapped_tags[i:i + tags_per_file] for i in range(0, len(unmapped_tags), tags_per_file)]
        for tags in tag_groups:
            f = Path(self._path).joinpath(uuid.uuid4().hex) # unique filename
            dataframe[tags].to_parquet(f)
            logging.info(f'Tags {tags} written to newly created file {f}')

        self.rebuild_vcb()

    def read(self, tags: List[str], begin: str=None, end: str=None) -> pd.DataFrame:
        '''Reads specified tags from the store'''
        tags_to_read = pd.Series(True, index=tags, name='Read')
        vcb_tags_to_read = self._vcb.join(tags_to_read, how='right').sort_values(by='Filename')
        vcb_tags_to_read['Existing'] = ~vcb_tags_to_read['Filename'].isna()

        existing_files = vcb_tags_to_read[vcb_tags_to_read['Existing']]['Filename'].unique().tolist()
        unmapped_tags = vcb_tags_to_read[~vcb_tags_to_read['Existing']].index.tolist()
        if len(unmapped_tags)>0:
            raise KeyError('No such tag in the store: ' + ', '.join(unmapped_tags))

        list_of_dataframes = []
        for f in existing_files:
            tags = vcb_tags_to_read[vcb_tags_to_read['Filename']==f].index.tolist()
            list_of_dataframes.append(pd.read_parquet(Path(self._path).joinpath(f), columns=tags))
            logging.debug(f'Read from file {f}. Tags: {tags}')
        d = pd.concat(list_of_dataframes, axis=1, join='outer', sort=True)
        if begin:
            d = d.loc[begin:]
        if end:
            d = d.loc[:end]
        return d

    def _get_xl_info(xl_path: str) -> Dict:
        wb = openpyxl.load_workbook(xl_path, read_only=True)
        xl_info = dict()
        for sh in wb.worksheets:
            sh.title

        return xl_info

    def update_from_excel_file(self, xl_path: str):
        list_of_dataframes = []
        logging.info(f'Reading {xl_path}')
        odct = pd.read_excel(xl_path, 
                             sheet_name=None,
                             index_col='Date')
        for sheetname, d in odct.items():
            d.index.name = 'Timestamp'
            d.drop('Time',axis=1, inplace=True)
            cols_to_float = (d.dtypes==np.int64)
            d.loc[:,cols_to_float]=d.loc[:,cols_to_float].astype('float64')
            list_of_dataframes.append(d)

        d = pd.concat(list_of_dataframes, axis=1, join='outer', sort=True)
        self.write(d)
        return

class StoreType(Enum):
    COLUMN_WISE = 1
    ROW_WISE = 2