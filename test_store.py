import warehouse
import pandas as pd

print('Hello!')
w = warehouse.Warehouse('./wh/')
print(w.path)
w.list_stores()
data10m = w.create_store('data10m.db')
w.list_stores()

print('Tables:')
print(data10m.get_tables())

data10m.list_tags()

dti = pd.DatetimeIndex(['2020-03-01 00:00:00', '2020-03-01 00:10:00',
               '2020-03-01 00:20:00', '2020-03-01 00:30:00'], name='Timestamp')
sample_df = pd.DataFrame({'Tag1': [0, 1, 2, 3],
                          'Tag2': [0, 10, 20, 30]},
                         index=dti)
data10m.low_level_write(sample_df, 'sample')