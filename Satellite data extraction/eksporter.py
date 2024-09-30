from Get_coordinates import fields,coordinates
import pandas as pd

print(fields)
print("______________")
print(coordinates)

dict = {'name': fields, 'coordinates': coordinates}
df = pd.DataFrame(dict)
df.to_csv('coords.csv')