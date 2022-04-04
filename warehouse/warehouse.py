r"""
 implemented as a folder of *parquet* files.
Designed for fast frequent reads and slow rare writes.

Warehouse is represented by a folder with parquet files. Each parquet file
holds pandas dataframe with DateTimeIndex and several columns (tags)
It is assumed that all dataframes in warehouse have identical sample rate.
The library is designed for working with databases containing hundreds or
several thousands of tags (columns) and hundreds of thousands rows.
Simple columnar storage for time-series
When dataframe is written to the warehouse, it is divided into portions of
10 tags per file (this default value can be changed).
Tag<->file mapping can be manually overridden when calling write(), or
warehouse can be fully reorganized later.

We cannot add or remove columns to file, so every write operation consists of:
- read whole file from disk,
- add data to file,
- write whole dataframe to disk.
This is why it's better to divide data into small portions

Use case:

import warehouse as wh
# Connect to warehouse
store = wh.Store(wh_path)
# List available tags:
store.vcb
# Read from warehouse
df = store.read(taglist=['2451FIC001.PV', '2451FIC202.PV'],
                begin='2021-03-25 00:00:00',
                end='2021-03-28 00:00:00')


# Write to warehouse
store.write(df)

# Convenient method for excel import
xl_path = 'C:\\Temp\\'
files = [f for f in os.scandir(xl_path) if f.is_file() and '.xlsx' in f.name]
for f in files:
    store.update_from_excel_file(f, resample_interval='10min')

"""
import logging
import logging.handlers
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastparquet import ParquetFile

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.TimedRotatingFileHandler('warehouse.log', when='D', backupCount=5)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(fh)


class Store:
    ''' Store object represents a folder which contains a number of parquet files'''

    def __init__(self, str_path: str = '//wh', tags_per_file=10):
        self.path = str_path
        self.tags_per_file = tags_per_file

        path = Path(str_path)
        if not path.is_dir():
            raise ValueError(f'Cannot connect to {path}')
        self._vcb = self._build_vcb()
        print('Succesfully connected to store.')

    @property
    def vcb(self):
        """Returns dataframe with index TagName and the following columns:
        Filename - where the tag is stored,
        Rows - number of rows,
        Begin - first timestamp,
        End - last timestamp. """
        return self._vcb

    @property
    def path(self):
        """Path to the folder of the store"""
        return self._path

    @path.setter
    def path(self, value: str):
        if os.path.isdir(value):
            self._path = value
        else:
            raise ValueError(f"Path {value} doesn't exist or is not a folder")

    @property
    def tags_per_file(self):
        """Number of tags to store in one file (only for newly created files)"""
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

    def _build_vcb(self):
        '''Rebuild vcb dataframe (tag <-> file mapping).
        Function returns resulting dataframe, but doesn't change self.vcb property '''
        files = [
            f
            for f in os.scandir(self._path)
            if f.is_file() and 'metadata' not in f.name
        ]
        tagname_list = []
        filename_list = []
        rows_list = []
        begin_list = []
        end_list = []

        for file in files:
            try:
                parquet_file = ParquetFile(file.path)
            except (FileNotFoundError, ValueError) as e:
                warnings.warn(f'{file.name} is not parquet file.')
                continue

            columns = parquet_file.info['columns']
            if 'Timestamp' in columns:
                columns.remove('Timestamp')
            else:
                warnings.warn(f'{file.name} doesnt contain Timestamp column')
                continue
            rows = parquet_file.info['rows']

            try:
                begin = str(parquet_file.statistics['min']['Timestamp'][0])[:16]
                end = str(parquet_file.statistics['max']['Timestamp'][0])[:16]
            except ValueError:
                logging.error(
                    'Unexpected error in rebuild_vcb: {}', sys.exc_info()[0])
                begin, end = '', ''

            for col in columns:
                tagname_list.append(col)
                filename_list.append(file.name)
                rows_list.append(rows)
                begin_list.append(begin)
                end_list.append(end)

        if tagname_list:
            vcb = pd.DataFrame(
                {
                    'Filename': filename_list,
                    'Rows': rows_list,
                    'Begin': begin_list,
                    'End': end_list,
                },
                index=pd.Series(tagname_list, name='TagName'),
            ).sort_index()
        else:
            vcb = pd.DataFrame(
                {'Filename': [], 'Rows': [], 'Begin': [], 'End': []},
                index=pd.Series([], name='TagName', dtype='object'),
            )
        return vcb

    def rebuild_vcb(self):
        '''Rebuild vcb dataframe (tag <-> file mapping). '''
        self._vcb = self._build_vcb()
        return self._vcb

    # TODO: strange warning from pandas
    # A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value instead
    # Where is it coming from?

    def _update_file(self, dataframe: pd.DataFrame, filename: str):
        file_path = Path(self.path).joinpath(filename)
        file_content = pd.read_parquet(file_path)

        _, _, new_overlaps, inconsistent_freq = check_time_range(
            file_content.index, dataframe.index
        )
        if new_overlaps:
            msg = (f'While updating file {filename} with tags ' +
                f'{dataframe.columns.tolist()}, timestamp overlap was detected.')
            logger.warning(msg)
            print(msg)
        if inconsistent_freq:
            msg = (f'While updating file {filename} with tags ' +
                f'{dataframe.columns.tolist()}, inconsistent sample rate was detected.')
            logger.error(msg)
            raise ValueError(msg)

        updated_frame = dataframe.combine_first(file_content)
        updated_frame.sort_index(inplace=True)
        updated_frame.to_parquet(file_path)
        logger.info(
            'Tags {} written to existing file {}', dataframe.columns.tolist(), filename
        )

    def write(self, dataframe: pd.DataFrame):
        '''Write dataframe tags to the store.
        Tag<->file mapping will be done automatically, but can also
        be provided in new_vcb dataframe (not supported yet)'''
        # TODO: support for new_vcb

        # TODO: think about the following situation:
        # first we write 1M data 00:05:00, 00:06:00, 00:07:00
        # then we write 1M data 00:05:10, 00:06:10, 00:07:10
        # when reading combined result it contains all
        # timestamps 00:05:00, 00:05:10, 00:06:00, 00:06:10
        # and some values are missing (filled with NaN).

        if not dataframe_is_valid(dataframe):
            raise ValueError('Please check if dataframe is valid.')

        # tags in dataframe:
        #   some already exist in the warehouse
        #       if file location is specified for existing tag and does not equal to its real
        #            location, then generate warning and exit
        #       otherwise save the tag to existing file location
        #   some are new
        #       if file location is specified, use it
        #       if not specified - generate new file for them
        #           or try to place new tags in existing files which have free space
        #           (this may be implemented later)

        # 1. Generate tag->file mapping: tag, file, existing
        tags_to_write = pd.Series(True, index=dataframe.columns, name='Write')
        vcb_tags_to_write = self._vcb.join(tags_to_write, how='right').sort_values(
            by='Filename'
        )
        vcb_tags_to_write['Existing'] = ~vcb_tags_to_write['Filename'].isna()

        existing_files = (
            vcb_tags_to_write[vcb_tags_to_write['Existing']]['Filename']
            .unique()
            .tolist()
        )
        unmapped_tags = vcb_tags_to_write[~vcb_tags_to_write['Existing']].index.tolist(
        )
        # 2. For each existing file,
        #       Call update_file(file, dataframe)
        for file in existing_files:
            tags = vcb_tags_to_write[vcb_tags_to_write['Filename'] == file].index.tolist(
            )
            self._update_file(dataframe[tags], file)

        # 3. For each non-existing file,
        #       Call dataframe.to_parquet()
        tags_per_file = 10
        tag_groups = [
            unmapped_tags[i: i + tags_per_file]
            for i in range(0, len(unmapped_tags), tags_per_file)
        ]
        for tags in tag_groups:
            file = Path(self._path).joinpath(
                uuid.uuid4().hex + '.parquet'
            )  # unique filename
            # TODO: check if whole tag_group is empty (contains no data).
            # This will result in wrong statistics in parquet file
            dataframe[tags].to_parquet(file)
            logger.info('Tags {} written to newly created file {}', tags, file)

        # Maybe its better to update only new/changed entries in vcb, rather than make full rebuild
        self.rebuild_vcb()

    def read(self, tags: List[str], begin: str = None, end: str = None) -> pd.DataFrame:
        '''Reads specified tags from the store'''

        tags_to_read = pd.Series(True, index=tags, name='Read')
        vcb_tags_to_read = self._vcb.join(tags_to_read, how='right').sort_values(
            by='Filename'
        )
        vcb_tags_to_read['Existing'] = ~vcb_tags_to_read['Filename'].isna()

        existing_files = (
            vcb_tags_to_read[vcb_tags_to_read['Existing']
                             ]['Filename'].unique().tolist()
        )
        unmapped_tags = vcb_tags_to_read[~vcb_tags_to_read['Existing']].index.tolist(
        )
        if len(unmapped_tags) > 0:
            raise KeyError('No such tag in the store: ' +
                           ', '.join(unmapped_tags))

        list_of_dataframes = []
        for filename in existing_files:
            file_tags = vcb_tags_to_read[
                vcb_tags_to_read['Filename'] == filename
            ].index.tolist()
            list_of_dataframes.append(
                pd.read_parquet(Path(self._path).joinpath(filename),
                                columns=file_tags)
            )
            logger.debug('Read from file {}. Tags: {}', filename, file_tags)
        dataframe = pd.concat(list_of_dataframes, axis=1,
                              join='outer', sort=True)
        if begin:
            dataframe = dataframe.loc[begin:]
        if end:
            dataframe = dataframe.loc[:end]
        # TODO: do we need resample? Is it possible to have different files in the store with
        # different samplerates?
        return dataframe[tags]

    def update_from_excel_file(self, xl_path: str, resample_interval: str = None):
        """ Loads the content of excel file to the store.
        Excel workbook may contain many sheets.
        Every sheet must have first two columns Date and Time"""

        # TODO: add timing

        logger.info('Reading {}', xl_path)
        # TODO: check if index column is called 'Date'
        # TODO: check for empty columns
        odct = pd.read_excel(xl_path, sheet_name=None, index_col='Date')
        for sheetname, dataframe in odct.items():
            if len(dataframe) == 0:
                logger.warning('Encountered empty sheet {}.', sheetname)
                continue
            dataframe.index.name = 'Timestamp'
            if 'Time' in dataframe.columns:
                dataframe.drop('Time', axis=1, inplace=True)
            # remove duplicated records (May appear when switching to daylight saving time)
            is_duplicated = dataframe.index.duplicated()
            n_duplicated = (
                is_duplicated.sum()
            )  # default keep='First', so old values are preserved
            if n_duplicated:
                logger.warning(
                    'While reading file {}, sheet {}, {} duplicated records were found.',
                    xl_path, sheetname, n_duplicated
                )
            dataframe = dataframe.loc[~is_duplicated]
            # convert int columns to float
            cols_to_float = dataframe.dtypes == np.int64
            dataframe.loc[:, cols_to_float] = dataframe.loc[:, cols_to_float].astype('float64')
            if resample_interval:
                dataframe = dataframe.resample(resample_interval).ffill()
            self.write(dataframe)

        return

    def delete(self, tags: List[str]):
        '''Delete specified tags from the store'''
        # TODO
        return

    def reorganize(self, new_path: str, new_vcb: pd.DataFrame):
        '''Move data to new storage with given tag<->file mapping'''
        # TODO:


# TODO: Do we need it here? Maybe ROW_WISE storage is easyly implemented with single sqlite table?
# ok, but we need 'update from excel' operation...
# class StoreType(Enum):
#    COLUMN_WISE = 1
#    ROW_WISE = 2


def dataframe_is_valid(dataframe: pd.DataFrame):
    '''Check if dataframe can be written to the store'''

    # must be a dataframe
    if not isinstance(dataframe, pd.DataFrame):
        print('Passed object is not a dataframe.')
        return False

    # must have datetime index
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        print('Dataframe index is not instance of DateTimeIndex')
        return False

    # index must be named 'Timestamp'
    if dataframe.index.name != 'Timestamp':
        print('Dataframe index must be named "Timestamp".')
        return False

    # index must be unique
    if dataframe.index.has_duplicates:
        print('Dataframe index has duplicates.')
        return False

    # index must be monotonically increasing
    if not dataframe.index.is_monotonic_increasing:
        print('Dataframe index not sorted.')
        return False

    return True


def get_freq(dti: pd.DatetimeIndex):
    '''Returns frequency of the index:
    index.freq if it is specified,
    index.inferred_freq if it can be computed,
    If it cannot be computed, tries to infer based on last 5 elements.'''

    if dti.freq:
        return dti.freq
    elif dti.inferred_freq:
        return dti.inferred_freq
    else:
        return dti[-5:].inferred_freq


def check_time_range(existing_index, new_index):
    """Check if new DatetimeIndex overlaps with existing and has the same freq"""
    exist_begin = existing_index[0]
    exist_end = existing_index[-1]
    new_begin = new_index[0]
    new_end = new_index[-1]

    if exist_begin > exist_end:
        raise ValueError('Begin of existing index is later then its end')
    if new_begin > new_end:
        raise ValueError('Begin of new index is later then its end')

    new_before, new_after, new_overlaps = False, False, False

    if new_begin > exist_end:
        new_after = True
    elif new_end < exist_begin:
        new_before = True
    else:
        new_overlaps = True

    existing_freq = get_freq(existing_index)
    new_freq = get_freq(new_index)
    inconsistent_freq = existing_freq and new_freq and (
        existing_freq != new_freq)

    return new_before, new_after, new_overlaps, inconsistent_freq
