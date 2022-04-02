""" Warehouse tests """

import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import warehouse as wh

WH_FOLDER = "..\\test_store\\"

class TestCreateWarehouse(unittest.TestCase):
    """Test creating of new warehouse or connect to existing warehouse"""
    def test_create_no_folder(self):
        """Create new warehouse, folder doesnt exist"""
        path = Path(WH_FOLDER)
        if path.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(WH_FOLDER)
        with self.assertRaises(
            ValueError, msg="Didn't raise error with non-existing folder"
        ):
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


    def test_create_folder_with_existing_other_files(self):
        """Create new warehouse, folder exists and filled with 
        some non-parquet files"""



class TestReadWrite(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(WH_FOLDER)
        if path.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(WH_FOLDER)
        path.mkdir()
        store = wh.Store(WH_FOLDER)

        nrows = 10
        dti = pd.date_range(
            start='2020-03-01 00:00:00',
            freq='10min',
            periods=nrows,
            name='Timestamp'
        )
        self.sample_df = pd.DataFrame(np.random.randn(nrows, 4), columns=list("ABCD"), index=dti)
        store.write(self.sample_df)
        return super().setUp()

    def test_read_existing(self):
        """Test read existing tag"""
        store = wh.Store(WH_FOLDER)
        dataframe = store.read(list("ABCD"))
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
