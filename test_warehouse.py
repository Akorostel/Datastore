""" Warehouse tests """

import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import warehouse as wh

wh_folder = "..\\test_store\\"


class TestCreateWarehouse(unittest.TestCase):
    def test_create_no_folder(self):
        """Create new warehouse, folder doesnt exist"""
        p = Path(wh_folder)
        if p.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(wh_folder)
        with self.assertRaises(
            ValueError, msg="Didn't raise error with non-existing folder"
        ):
            store = wh.Store(wh_folder)

    def test_create_empty_folder(self):
        p = Path(wh_folder)
        if p.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(wh_folder)
        p.mkdir()
        store = wh.Store(wh_folder)
        self.assertTrue(store.vcb.empty, "Returned non-empty vcb just after creation")

    def test_create_folder_with_existing_parquet_files(self):
        pass

    def test_create_folder_with_existing_other_files(self):
        pass


class TestReadWrite(unittest.TestCase):
    def setUp(self) -> None:
        p = Path(wh_folder)
        if p.is_dir():
            print("Deleting old folder...")
            shutil.rmtree(wh_folder)
        p.mkdir()
        store = wh.Store(wh_folder)

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
        store = wh.Store(wh_folder)
        df = store.read(list("ABCD"))
        print(self.sample_df)
        print(df)
        self.assertTrue(self.sample_df.equals(df))
        pass

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
