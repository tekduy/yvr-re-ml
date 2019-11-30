# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:12:59 2019

@author: Donald
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from geopy.geocoders import Nominatim
import re
from datetime import date, datetime
import matplotlib.pyplot as plt
import seaborn

#212 838 HAMILTON STREET
address_pattern = re.compile(r'^(\S+) \d+ \S+ \S+')
house_number_pattern = re.compile(r'^\S+')
penthouse_pattern = re.compile(r'^[PH]')
other_house_pattern = re.compile(r'[^0-9]')
apartment_pattern = re.compile(r'^[0-9]+[a-zA-Z]+$')

def read_csv():
    csvs = ['2014-12.csv',
            '2014-23.csv',
            '2015-1.csv',
            '2015-2.csv',
            '2015-3.csv',
            '2016-1.csv',
            '2016-2.csv',
            '2016-3.csv',
            '2017-1.csv',
            '2017-2.csv',
            '2017-3.csv',
            '2018-1.csv',
            '2018-23.csv',
            '2019-123.csv']

    data = [pd.read_csv(csv) for csv in csvs]

    return data

def concatenate_csvs(data):
    result = pd.concat([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11],
            data[12], data[13]])

    return result


def nulls_to_zeros(d):
    if pd.isnull(d):
        return 0
    else:
        return d

def get_HouseNumber(d):
    m = address_pattern.match(d)
    if m:
        house_number = m.group(1)
        h = penthouse_pattern.match(house_number) #Penthouse = 5000
        other = other_house_pattern.match(house_number) #Other type of house number = 1000
        a = apartment_pattern.match(house_number)
        if h:
            return 5000
        elif other:
            return 1000
        elif a:
            return 1000
        else:
            return house_number

def format_GeoPy(d):
    new_address = re.sub(house_number_pattern, '', d) #Remove house number
    new_address = new_address.lstrip()
    new_address = new_address + ', VANCOUVER' #Add Vancouver to address
    return new_address

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

def to_datetime(d):
    return pd.to_datetime(d)

def to_timestamp(d):
    return d.timestamp()

def to_float(d):
    return float(d)

def fix_maint_fees(x, y):
    if pd.isnull(y):
        return (x * 0.4)
    else:
        return y

def main():
    data = read_csv()
    result = concatenate_csvs(data)
    #result = data[13]

    #Rename columns
    result = result.rename(index=str, columns={"S/A": "SubArea",
                                               "DOM": "DaysOnMarket",
                                               "Tot BR": "Bedrooms",
                                               "Tot Baths": "Bathrooms",
                                               "TotFlArea": "FloorArea",
                                               "Yr Blt": "YearBuilt",
                                               "TotalPrkng": "Parking",
                                               "StratMtFee": "MaintenanceFees",
                                               "SP Sqft": "SalePricePerSquareFoot",
                                               "TotalPrkng": "Parking"})

    #Remove $ and , from prices
    result['Price'] = result['Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['MaintenanceFees'] = result['MaintenanceFees'].astype(str).map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['SalePricePerSquareFoot'] = result['SalePricePerSquareFoot'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['FloorArea'] = result['FloorArea'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['List Price'] = result['List Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    result['Sold Price'] = result['Sold Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    #print(result)

    #Convert Parking to 0 if null
    result['Parking'] = result['Parking'].apply(nulls_to_zeros)

    #Get HouseNumber
    #result['HouseNumber'] = result['Address'].apply(get_HouseNumber)

    #Convert to GeoPy format
    #result['Address'] = result['Address'].apply(format_GeoPy)

    #Convert VwSpecify to uppercase
    result['VwSpecify'] = result['VwSpecify'].astype(str).apply(uppercase)
    result['VwSpecify'] = result['VwSpecify'].replace('NAN', '')

    #Convert VwSpecify to -1,0,1,2
    result['ValueOfView'] = result.apply(lambda x: convert_VwSpecify(x['View'], x['VwSpecify']), axis=1)

    #Convert SubArea to Numeric Value (VVWCB = Coal Harbour = 1, VVWDT = Downtown = 2, VVWWE = West End = 3, VVWYA = Yaletown = 4)
    result['ValueOfSubArea'] = result['SubArea']
    result['ValueOfSubArea'] = result['ValueOfSubArea'].replace('VVWCB', '1')
    result['ValueOfSubArea'] = result['ValueOfSubArea'].replace('VVWDT', '2')
    result['ValueOfSubArea'] = result['ValueOfSubArea'].replace('VVWWE', '3')
    result['ValueOfSubArea'] = result['ValueOfSubArea'].replace('VVWYA', '4')

    #Change Locker to numeric - Yes = 1, No = 0, Null = 0
    result['Locker'] = result['Locker'].replace('Yes', '1')
    result['Locker'] = result['Locker'].replace('No', '0')
    result['Locker'] = result['Locker'].apply(nulls_to_zeros)

    #Change Sold Date to datetime and timestamp
    result['Sold Date'] = result['Sold Date'].apply(to_datetime)
    result['Sold Timestamp'] = result['Sold Date'].apply(to_timestamp)

    #Convert all columns to float
    result['ValueOfSubArea'] = result['ValueOfSubArea'].apply(to_float)
    result['DaysOnMarket'] = result['DaysOnMarket'].apply(to_float)
    result['Bedrooms'] = result['Bedrooms'].apply(to_float)
    result['Bathrooms'] = result['Bathrooms'].apply(to_float)
    result['FloorArea'] = result['FloorArea'].apply(to_float)
    result['YearBuilt'] = result['YearBuilt'].apply(to_float)
    result['Locker'] = result['Locker'].apply(to_float)
    result['Parking'] = result['Parking'].apply(to_float)
    result['Sold Timestamp'] = result['Sold Timestamp'].apply(to_float)
    result['ValueOfView'] = result['ValueOfView'].apply(to_float)
    result['List Price'] = result['List Price'].apply(to_float)
    result['Sold Price'] = result['Sold Price'].apply(to_float)
    result['MaintenanceFees'] = result['MaintenanceFees'].apply(to_float)
    result['SalePricePerSquareFoot'] = result['SalePricePerSquareFoot'].apply(to_float)

    #Change nulls to an estimated maintenance fee
    result['MaintenanceFees'] = result.apply(lambda x: fix_maint_fees(x['FloorArea'], x['MaintenanceFees']), axis=1)

    #Scale-down List Price and Sold Price
    result['List Price'] = result['List Price']/10000
    result['Sold Price'] = result['Sold Price']/10000
    result['MaintenanceFees'] = result['MaintenanceFees']/100
    result['SalePricePerSquareFoot'] = result['SalePricePerSquareFoot']/100

    # print(result)
    result.to_csv('2014-2019-cleaned.csv', index=False, header=True)

    X = result[['ValueOfSubArea', 'DaysOnMarket',
                'Bedrooms', 'Bathrooms', 'FloorArea', 'YearBuilt', 'Age',
                'Locker', 'Parking', 'MaintenanceFees',
                'List Price', 'Sold Timestamp', 'ValueOfView']].values
    y = result['Sold Price'].values

    # print(result['Address'])
    # Convert address to lat and long
    # geolocator = Nominatim(user_agent="Akshay", timeout=60, country_bias='CA')
    # # location = geolocator.geocode()
    # try:
    #     location = result.apply(lambda row: geolocator.geocode(row['Address']), axis=1)
    #     print(location)
    # except GeocoderTimedOut as e:
    #     print(e)
    # print((location.latitude, location.longitude))

    #print(X)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)

    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(18, 16),
        activation='logistic', solver='lbfgs'))
    model.fit(X_train, y_train)
    print("Model score: ", model.score(X_valid, y_valid))
    
    #Perform Prediction
    predict_price = pd.read_csv('2019-predict.csv')

    #Rename columns
    predict_price = predict_price.rename(index=str, columns={"S/A": "SubArea",
                                               "DOM": "DaysOnMarket",
                                               "Tot BR": "Bedrooms",
                                               "Tot Baths": "Bathrooms",
                                               "TotFlArea": "FloorArea",
                                               "Yr Blt": "YearBuilt",
                                               "TotalPrkng": "Parking",
                                               "StratMtFee": "MaintenanceFees",
                                               "SP Sqft": "SalePricePerSquareFoot",
                                               "TotalPrkng": "Parking"})

    #Remove $ and , from prices
    predict_price['Price'] = predict_price['Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    predict_price['MaintenanceFees'] = predict_price['MaintenanceFees'].astype(str).map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    predict_price['SalePricePerSquareFoot'] = predict_price['SalePricePerSquareFoot'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    predict_price['FloorArea'] = predict_price['FloorArea'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    predict_price['List Price'] = predict_price['List Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    predict_price['Sold Price'] = predict_price['Sold Price'].map(lambda x: x.lstrip('$')).replace(',', '', regex=True)
    #print(predict_price)

    #Convert Parking to 0 if null
    predict_price['Parking'] = predict_price['Parking'].apply(nulls_to_zeros)

    #Get HouseNumber
    #predict_price['HouseNumber'] = predict_price['Address'].apply(get_HouseNumber)

    #Convert to GeoPy format
    #predict_price['Address'] = predict_price['Address'].apply(format_GeoPy)

    #Convert VwSpecify to uppercase
    predict_price['VwSpecify'] = predict_price['VwSpecify'].astype(str).apply(uppercase)
    predict_price['VwSpecify'] = predict_price['VwSpecify'].replace('NAN', '')

    #Convert VwSpecify to -1,0,1,2
    predict_price['ValueOfView'] = predict_price.apply(lambda x: convert_VwSpecify(x['View'], x['VwSpecify']), axis=1)

    #Convert SubArea to Numeric Value (VVWCB = Coal Harbour = 1, VVWDT = Downtown = 2, VVWWE = West End = 3, VVWYA = Yaletown = 4)
    predict_price['ValueOfSubArea'] = predict_price['SubArea']
    predict_price['ValueOfSubArea'] = predict_price['ValueOfSubArea'].replace('VVWCB', '1')
    predict_price['ValueOfSubArea'] = predict_price['ValueOfSubArea'].replace('VVWDT', '2')
    predict_price['ValueOfSubArea'] = predict_price['ValueOfSubArea'].replace('VVWWE', '3')
    predict_price['ValueOfSubArea'] = predict_price['ValueOfSubArea'].replace('VVWYA', '4')

    #Change Locker to numeric - Yes = 1, No = 0, Null = 0
    predict_price['Locker'] = predict_price['Locker'].replace('Yes', '1')
    predict_price['Locker'] = predict_price['Locker'].replace('No', '0')
    predict_price['Locker'] = predict_price['Locker'].apply(nulls_to_zeros)

    #Change Sold Date to datetime and timestamp
    predict_price['Sold Date'] = predict_price['Sold Date'].apply(to_datetime)
    predict_price['Sold Timestamp'] = predict_price['Sold Date'].apply(to_timestamp)

    #Convert all columns to float
    predict_price['ValueOfSubArea'] = predict_price['ValueOfSubArea'].apply(to_float)
    predict_price['DaysOnMarket'] = predict_price['DaysOnMarket'].apply(to_float)
    predict_price['Bedrooms'] = predict_price['Bedrooms'].apply(to_float)
    predict_price['Bathrooms'] = predict_price['Bathrooms'].apply(to_float)
    predict_price['FloorArea'] = predict_price['FloorArea'].apply(to_float)
    predict_price['YearBuilt'] = predict_price['YearBuilt'].apply(to_float)
    predict_price['Locker'] = predict_price['Locker'].apply(to_float)
    predict_price['Parking'] = predict_price['Parking'].apply(to_float)
    predict_price['Sold Timestamp'] = predict_price['Sold Timestamp'].apply(to_float)
    predict_price['ValueOfView'] = predict_price['ValueOfView'].apply(to_float)
    predict_price['List Price'] = predict_price['List Price'].apply(to_float)
    predict_price['Sold Price'] = predict_price['Sold Price'].apply(to_float)
    predict_price['MaintenanceFees'] = predict_price['MaintenanceFees'].apply(to_float)
    predict_price['SalePricePerSquareFoot'] = predict_price['SalePricePerSquareFoot'].apply(to_float)

    #Change nulls to an estimated maintenance fee
    predict_price['MaintenanceFees'] = predict_price.apply(lambda x: fix_maint_fees(x['FloorArea'], x['MaintenanceFees']), axis=1)

    #Scale-down List Price and Sold Price
    predict_price['List Price'] = predict_price['List Price']/10000
    predict_price['Sold Price'] = predict_price['Sold Price']/10000
    predict_price['MaintenanceFees'] = predict_price['MaintenanceFees']/100
    predict_price['SalePricePerSquareFoot'] = predict_price['SalePricePerSquareFoot']/100

    X_predict = predict_price[['ValueOfSubArea', 'DaysOnMarket',
                               'Bedrooms', 'Bathrooms', 'FloorArea', 'YearBuilt', 'Age',
                               'Locker', 'Parking', 'MaintenanceFees',
                               'List Price', 'Sold Timestamp', 'ValueOfView']].values
    
    y_predict = model.predict(X_predict)
    y_predict = y_predict
    
    predict_price['Predicted Price'] = y_predict
    predict_price['Sold Price'] = predict_price['Sold Price']
    predict_price['Predicted/Sold'] = predict_price['Predicted Price']/predict_price['Sold Price']
    predict_price['Residuals']= predict_price['Sold Price'] - predict_price['Predicted Price']
    header = ["Address", "Sold Price", "Predicted Price", "Residuals", "Predicted/Sold"]
    predict_price.to_csv('output_prediction.csv', columns=header, index=False)
    
    difference_mean = predict_price['Predicted/Sold'].mean()
    residual_mean = predict_price['Residuals'].mean()
    print("Mean of Predicted Price divided by Sold Price: ", difference_mean)
    print("On average, the Predicted Price proportional to the Sold Price by:", difference_mean*100,"%")
    print("Mean of Residuals: ", residual_mean)
    
    seaborn.set()
    plt.title('Sold Price and Predicted Price')
    plt.xlabel('Property')
    plt.ylabel('Price * 10000')
    sold_price = plt.scatter(range(100), predict_price['Sold Price'], color='blue', s=8)
    predicted_price = plt.scatter(range(100), predict_price['Predicted Price'], color='red', s=8)
    plt.legend((sold_price, predicted_price), ('Sold Price', 'Predicted Price'))
    #plt.show()
    
    list_price = input("Please enter list price: ")
    list_price = float(list_price)/10000
    floor_area = input("Please enter floor area (in square feet): ")
    floor_area = float(floor_area)
    bedrooms = input("Please enter number of bedrooms: ")
    bedrooms = float(bedrooms)
    bathrooms = input("Please enter number of bathrooms: ")
    bathrooms = float(bathrooms)
    parking = input("Please enter number of parking stalls: ")
    parking = float(parking)
    locker = input("Please enter number of storage lockers: ")
    locker = float(locker)
    maintenance_fees = input("Please enter maintenance fees in dollars per month: ")
    maintenance_fees = float(maintenance_fees)/100
    year_built = input("Please enter year built: ")
    age = 2019 - float(year_built)
    age = float(age)
    days_on_market = input("Please enter number of days the listing has been on the market: ")
    days_on_market = float(days_on_market)
    sub_area = input("Please enter sub area (Coal harbour = 1, Downtown = 2, West End = 3, Yaletown = 4): ")
    sub_area = float(sub_area)
    view = input("Please enter value for view (No view = 0, view but other = 1, mountains or city = 2, any kind of water view = 3): ")
    view = float(view)
    today = datetime.now()
    sold_timestamp = datetime.timestamp(today)
    #sale_price_sqfoot = list_price/floor_area

    X_predict = [[sub_area, days_on_market, bedrooms, bathrooms, floor_area, year_built,
                 age, locker, parking, maintenance_fees, list_price,
                 sold_timestamp, view]]

    y_predict = model.predict(X_predict)

    print("Your sold price should be: ", y_predict*10000)

if __name__ == '__main__':
    main()
