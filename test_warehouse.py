""" Warehouse tests """

import unittest
import warehouse as wh
from pathlib import Path
import shutil

wh_folder = '..\\test_store\\'

class TestWarehouse(unittest.TestCase):

    def test_create_no_folder(self):
        '''Create new warehouse, folder doesnt exist'''
        p = Path(wh_folder)
        if p.is_dir():
            print('Deleting old folder...')
            shutil.rmtree(wh_folder)
        with self.assertRaises(ValueError):
            store = wh.Store(wh_folder)
    
#dti = pd.DatetimeIndex(['2020-03-01 00:00:00', '2020-03-01 00:10:00',
#               '2020-03-01 00:20:00', '2020-03-01 00:30:00'], name='Timestamp')
#sample_df = pd.DataFrame({'Tag1': [0, 1, 2, 3],
#                          'Tag2': [0, 10, 20, 30]},
#                         index=dti)
    

if __name__ == '__main__':
    unittest.main()