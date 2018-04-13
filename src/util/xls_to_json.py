'''
Created on Apr 2, 2018

@author: mzehlike
'''

"""XLS -> json converter
first:
  $ pip install xlrd
then:
  $ cat in.xls
date, temp, pressure
Jan 1, 73, 455
Jan 3, 72, 344
Jan 7, 52, 100
convert:
  $ python xls_to_json.py in.xls Sheet1 out.json
finally:
  $ cat out.json
{
  'data': [
    {'date': 'Jan 1', 'temp': 73, 'pressure': 455},
    {'date': 'Jan 3', 'temp': 72, 'pressure': 344},
    {'date': 'Jan 7', 'temp': 52, 'pressure': 100},
  ]
}
"""

import json
import sys

import xlrd


workbook = xlrd.open_workbook(sys.argv[1])
worksheet = workbook.sheet_by_name(sys.argv[2])

data = []
keys = [v.value for v in worksheet.row(0)]
for row_number in range(worksheet.nrows):
    if row_number == 0:
        continue
    row_data = {}
    for col_number, cell in enumerate(worksheet.row(row_number)):
        row_data[keys[col_number]] = cell.value
    data.append(row_data)

with open(sys.argv[3], 'w') as json_file:
    json_file.write(json.dumps({'data': data}))


