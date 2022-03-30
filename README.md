Simple columnar storage for time-series implemented as a folder of *parquet* files.
Designed for fast frequent reads and slow rare writes.

Warehouse is represented by a folder with parquet files. Each parquet file holds pandas dataframe with DateTimeIndex and several columns (tags)
It is assumed that all dataframes in warehouse have identical sample rate.
The library is designed for working with databases containing hundreds or several thousands of tags (columns) and hundreds of thousands rows.

When dataframe is written to the warehouse, it is divided into portions of 10 tags per file (this default value can be changed). 
Tag<->file mapping can be manually overridden when calling write(), or warehouse can be fully reorganized later.

We cannot add or remove columns to file, so every write operation consists of:
- read whole file from disk,
- add data to file,
- write whole dataframe to disk.
This is why it's better to divide data into small portions

Use case:
```python
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


```