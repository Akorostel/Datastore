""" 
Warehouse is a folder with parquet files. Each parquet file holds pandas dataframe with DateTimeIndex and several columns (tags)
It is assumed that all dataframes in warehouse have identical sample rate.
The library is designed for working with databases containing hundreds or several thousands of tags (columns) and hundreds of thousands rows.

When dataframe is written to the warehouse, it is divided into portions of 10 tags per file (this default value can be changed). 
Tag<->file mapping can be manually overridden when calling write(), or warehouse can be fully rebuilt later.

We cannot add or remove columns to file, so every write operation consists of:
- read whole file from disk,
- add data to file,
- write whole dataframe to disk.
This is why it's better to divide data to small portions

Every folder has 'builtin' parquet file named vcb, which contains
list of available tags, tag <-> file mapping, and (under question) other metadata (description, units, ...)

Use case:
import warehouse as wh
# Connect to warehouse 
store = wh.Store(wh_path)
# List available tags:
store.vcb 
# Read from warehouse 
df = store.read(taglist=['2101FIC201.PV', '2101FIC205.PV'],
                begin='2021-03-25 00:00:00',
                end='2021-03-28 00:00:00')
# Write to warehouse
store.write(df)
"""

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
import sys

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.TimedRotatingFileHandler('warehouse.log', when='D', backupCount=5)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(fh)

class Store:
    ''' Store object represents a folder which contains a number of parquet files'''

    def __init__(self, path: str = '//wh', tags_per_file=10):
        self.path = path
        self.tags_per_file = tags_per_file

        p = Path(path)
        if not p.is_dir():
            raise ValueError(f'Cannot connect to {p}')
        self._vcb = self.build_vcb()
        print('Succesfully connected to store.')

    @property
    def vcb(self):
        return self._vcb
        
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: str):
        if os.path.isdir(value):
            self._path = value
        else:
            raise ValueError(f"Path {value} doesn't exist or is not a folder")

    @property
    def tags_per_file(self):
        return self._tags_per_file

    @tags_per_file.setter
    def tags_per_file(self, value: int):
        if value < 1:
            value = 1
            warnings.warn('tags_per_file set to 1.')
        elif value > 1000:
            value = 1000
            warnings.warn('tags_per_file set to 1000.')
        self._tags_per_file = value    

    def build_vcb(self):
        '''Rebuild vcb dataframe (tag <-> file mapping) '''
        files = [f for f in os.scandir(self._path) if f.is_file() and 
                                                      'metadata' not in f.name]
        tagname_list = []
        filename_list = []
        rows_list = []
        begin_list = []
        end_list = []

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
            rows = pf.info['rows']
            
            try:
                begin = str(pf.statistics['min']['Timestamp'][0])[:16]
                end = str(pf.statistics['max']['Timestamp'][0])[:16]
            except:
                logging.error(f'Unexpected error in rebuild_vcb: {sys.exc_info()[0]}')
                begin, end = '', ''
            
            for c in columns:
                tagname_list.append(c)
                filename_list.append(f.name)
                rows_list.append(rows)
                begin_list.append(begin)
                end_list.append(end)

        if tagname_list:
            vcb = pd.DataFrame({'Filename': filename_list, 
                                  'Rows': rows_list,
                                  'Begin': begin_list,
                                  'End': end_list}, 
                                  index=pd.Series(tagname_list, name='TagName')).sort_index()
        else:
            vcb = pd.DataFrame({'Filename': [], 'Rows': [], 'Begin': [], 'End': []}, 
                                     index=pd.Series([], name='TagName', dtype='object'))
        return vcb

    def rebuild_vcb():
        self._vcb = build_vcb()
        return self._vcb
    
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
        
    def check_time_range(exist_begin, exist_end, new_begin, new_end):
        #TODO:
        if isinstance(exist_begin, str):
            exist_begin = pd.Timestamp(exist_begin)
        elif not isinstance(exist_begin, pd.Timestamp):
            raise ValueError(f'exist_begin {exist_begin} is not a valid timestamp')
            
        new_before, new_after, new_partly_overlaps = False, False, False    
        return
    
    #TODO: strange warning from pandas
    # A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value instead
    # Where is it coming from?
    
    def update_file(self, dataframe: pd.DataFrame, filename: str):
        # TODO: check if time ranges of existing and new dataframes intersect, and generate warning 
        # TODO: check if samplerate is different, and generate warning or error

        p = Path(self.path).joinpath(filename)
        file_content = pd.read_parquet(p)
        #updated_frame = pd.concat([file_content, dataframe])
        #is_duplicated = updated_frame.index.duplicated()
        #n_duplicated = is_duplicated.sum() # default keep='First', so old values are preserved
        #TODO: this is wrong warning
        #if you have a file of 10 tags, and you first update 5 tags, and in the second run you update another 5 tags,
        # then you;ll get a warning on the a lot of duplicated records on the second run
        #if n_duplicated:
        #    logger.warning(f'While updating file {filename}, {n_duplicated} duplicated records were found.')
        #updated_frame = updated_frame.loc[~is_duplicated]
        updated_frame = dataframe.combine_first(file_content)
        updated_frame.sort_index(inplace=True)
        updated_frame.to_parquet(p)
        logger.info(f'Tags {dataframe.columns.tolist()} written to existing file {filename}')

    def write(self, dataframe: pd.DataFrame, new_vcb: pd.DataFrame = None):
        '''Write dataframe tags to the store.
        Tag<->file mapping will be done automatically, but can also
        be provided in new_vcb dataframe (not supported yet)'''
        #TODO: support for new_vcb
        #TODO: check if dataframe is valid
        #TODO: check if dataframe's samplerate is compatible with samplerate of the store
        
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

        # 3. For each non-existing file,
        #       Call dataframe.to_parquet()
        tags_per_file = 10
        tag_groups = [unmapped_tags[i:i + tags_per_file] for i in range(0, len(unmapped_tags), tags_per_file)]
        for tags in tag_groups:
            f = Path(self._path).joinpath(uuid.uuid4().hex + '.parquet') # unique filename
            dataframe[tags].to_parquet(f)
            logger.info(f'Tags {tags} written to newly created file {f}')

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
            file_tags = vcb_tags_to_read[vcb_tags_to_read['Filename']==f].index.tolist()
            list_of_dataframes.append(pd.read_parquet(Path(self._path).joinpath(f), columns=file_tags))
            logger.debug(f'Read from file {f}. Tags: {file_tags}')
        d = pd.concat(list_of_dataframes, axis=1, join='outer', sort=True)
        if begin:
            d = d.loc[begin:]
        if end:
            d = d.loc[:end]
        #TODO: do we need resample? Is it possible to have different files in the store with
        # different samplerates?
        return d[tags]

    def _get_xl_info(xl_path: str) -> Dict:
        wb = openpyxl.load_workbook(xl_path, read_only=True)
        xl_info = dict()
        for sh in wb.worksheets:
            sh.title # get max_col, max_row takes too much time

        return xl_info

    def update_from_excel_file(self, xl_path: str, resample_interval:str=None):
        #list_of_dataframes = []
        logger.info(f'Reading {xl_path}')
        #TODO: check if index column is called 'Date'
        odct = pd.read_excel(xl_path, 
                             sheet_name=None,
                             index_col='Date')
        for sheetname, d in odct.items():
            d.index.name = 'Timestamp'
            if 'Time' in d.columns:
                d.drop('Time',axis=1, inplace=True)
            # remove duplicated records (May appear when switching to daylight saving time)
            is_duplicated = d.index.duplicated()
            n_duplicated = is_duplicated.sum() # default keep='First', so old values are preserved
            if n_duplicated:
                logger.warning(f'While reading file {xl_path}, sheet {sheetname}, {n_duplicated} duplicated records were found.')
            d = d.loc[~is_duplicated]
            # convert int columns to float
            cols_to_float = (d.dtypes==np.int64)
            d.loc[:,cols_to_float]=d.loc[:,cols_to_float].astype('float64')
            if resample_interval:
                d = d.resample(resample_interval).ffill()
            self.write(d)
            #list_of_dataframes.append(d)
        #d = pd.concat(list_of_dataframes, axis=1, join='outer', sort=True)
        #self.write(d)
        return

    def reorganize(self, new_path: str, new_vcb: pd.DataFrame):
        '''Move data to new storage with given tag<->file mapping'''
        #TODO:
        pass
    
# TODO: Do we need it here? Maybe ROW_WISE storage is easyly implemented with single sqlite table?
# ok, but we need 'update from excel' operation...
class StoreType(Enum):
    COLUMN_WISE = 1
    ROW_WISE = 2