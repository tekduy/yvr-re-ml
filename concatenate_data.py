# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:12:59 2019

@author: Donald
"""

import pandas as pd

def read_csv():
    data1 = pd.read_csv('2014-12.csv')
    data2 = pd.read_csv('2014-23.csv')
    data3 = pd.read_csv('2015-1.csv')
    data4 = pd.read_csv('2015-2.csv')
    data5 = pd.read_csv('2015-3.csv')
    data6 = pd.read_csv('2016-1.csv')
    data7 = pd.read_csv('2016-2.csv')
    data8 = pd.read_csv('2016-3.csv')
    data9 = pd.read_csv('2017-1.csv')
    data10 = pd.read_csv('2017-2.csv')
    data11 = pd.read_csv('2017-3.csv')
    data12 = pd.read_csv('2018-1.csv')
    data13 = pd.read_csv('2018-23.csv')
    data14 = pd.read_csv('2019-123.csv')

    return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14

def uppercase(d):
    return d.upper()

def convert_VwSpecify(x,y):
    #Adapted from: https://datascience.stackexchange.com/questions/56668/pandas-change-value-of-a-column-based-another-column-condition
    if x == 'Yes' and pd.isnull(y):
        return 0
    elif pd.isnull(x):
        return 0
    elif x == 'No':
        return -1
    elif x == 'Yes':
        if 'MOUNTAIN' in y:
            return 2
        elif 'CITY' in y:
            return 2
        elif 'WATER' in y:
            return 3
        elif 'HARBOUR' in y:
            return 3
        elif 'OCEAN' in y:
            return 3
        else:
            return 1


def main():
    # data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14 = read_csv()
    # result = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14])
    data14 = read_csv()
    result = data14
    #Remove $ and , from prices
    result['Price'] = result['Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['StratMtFee'] = result['StratMtFee'].astype(str).map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['SP Sqft'] = result['SP Sqft'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['List Price'] = result['List Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['Sold Price'] = result['Sold Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    #print(result)
    # Change NAN to 0
    result['VwSpecify'] = result['VwSpecify'].replace('NAN', '')

    #Convert VwSpecify to uppercase
    result['VwSpecify'] = result['VwSpecify'].astype(str).apply(uppercase)
    result['VwSpecify'] = result['VwSpecify'].replace('NAN', '')

    #Convert VwSpecify to -1,0,1,2
    result['VwSpecify_Numeric'] = result.apply(lambda x: convert_VwSpecify(x['View'], x['VwSpecify']), axis=1)
    result.rename(index=str, columns={"S/A": "SubArea", "DOM": "DaysOnMarket", "Tot BR": "Bedrooms", "Tot Baths": "Bathrooms", "TotFlArea": "FloorArea", "YrBlt": "YearBuilt", "TotalPrkng": "Parking", "StratMtFee": "MaintenanceFees", "SP Sqft": "SalePricePerSquareFoot", "TotalPrkng": "Parking", "VwSpecify_Numeric": "ValueOfView"})

    X = resut[['Address', 'SubArea', 'DaysOnMarket', 'Bedrooms', 'Bathrooms', 'FloorArea', 'YearBuilt', 'Age', 'Locker', 'Parking', 'MaintenanceFees', 'SalePricePerSquareFoot', 'List Price', 'Sold Date', 'ValueOfView']]
    y = resut['Sold Price'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)

    print(result)
    result.to_csv('2014-2019-cleaned.csv', index=False, header=True)

if __name__ == '__main__':
    main()