import warehouse

print('Hello!')
w = warehouse.Warehouse('./wh/')
print(w.path)
w.list_stores()
data10m = w.create_store('data10m.db')
w.list_stores()
