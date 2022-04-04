""" Warehouse tests """

import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import warehouse.warehouse as wh

WH_FOLDER = "..\\test_store\\"

def get_sample_dataframe(n_tags=10, n_samples=1000, random=True):
    """Returns sample dataframe with datetime index, containing either
    random numbers (if random==True) or arithmetic progression (if random==False)"""
    dt_index = pd.date_range(
        start='2022-01-01 00:00:00',
        periods=n_samples,
        freq='1min',
        name='Timestamp'
    )
    column_names = [f'1000FIC{i:05d}.PV' for i in range(n_tags)]
    if random:
        values = np.random.randn(n_samples, n_tags)
    else:
        values = [list(range(0 + i, n_samples + i)) for i in range(n_tags)]
    return pd.DataFrame(values, columns=column_names, index=dt_index)

class TestCreateWarehouse(unittest.TestCase):
    """Test creating of new warehouse or connect to existing warehouse"""
    def test_create_no_folder(self):
        """Create new warehouse, folder doesnt exist"""
        path = Path(WH_FOLDER)
        if path.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(WH_FOLDER)
        with self.assertRaises(ValueError, msg="Didn't raise error with non-existing folder"):
            _ = wh.Store(WH_FOLDER)

    def test_create_empty_folder(self):
        """Create new warehouse, empty folder exists"""
        path = Path(WH_FOLDER)
        if path.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(WH_FOLDER)
        path.mkdir()
        store = wh.Store(WH_FOLDER)
        self.assertTrue(store.vcb.empty, "Returned non-empty vcb just after creation")

    def test_create_folder_with_existing_parquet_files(self):
        """Create new warehouse, folder exists and filled with
        some parquet files"""
        path = Path(WH_FOLDER)
        if path.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(WH_FOLDER)
        path.mkdir()

        # Create test dataframe and save it to parquet file
        n_tags = 10
        get_sample_dataframe(n_tags=n_tags).to_parquet(path.joinpath('test01.parquet'))

        store = wh.Store(WH_FOLDER)
        print(store.vcb)
        self.assertEqual(store.vcb.shape[0], n_tags)

    def test_create_folder_with_existing_other_files(self):
        """Create new warehouse, folder exists and filled with
        some non-parquet files"""

class TestReadWrite(unittest.TestCase):
    """Test read and write operations"""
    def setUp(self) -> None:
        path = Path(WH_FOLDER)
        if path.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(WH_FOLDER)
        path.mkdir()
        store = wh.Store(WH_FOLDER)

        self.sample_df = get_sample_dataframe()
        self.tag_names = self.sample_df.columns.to_list()
        store.write(self.sample_df)
        return super().setUp()

    def test_read_existing(self):
        """Test read existing tag"""
        store = wh.Store(WH_FOLDER)
        dataframe = store.read(self.tag_names)
        print(self.sample_df)
        print(dataframe)
        self.assertTrue(self.sample_df.equals(dataframe))

    def test_write_read_random(self):
        pass

    def test_write_read_csv(self):
        pass

    def test_many_writes_single_read(self):
        pass

    def test_single_write_many_reads(self):
        pass


if __name__ == "__main__":
    print('started')
    unittest.main()
