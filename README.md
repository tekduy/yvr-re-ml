# CMPT 353: Final Project
# Predicting Real Estate Sales Prices through Machine Learning
Members:

Akshay Agrawal (301153611)

Tek Donald Duy (301223216)

# Problem
Realtors make their living guiding people through what is most likely the largest single purchase of their lives. They add value in several ways, however, with the plethora of research tools available for buyers, buyers are more informed and less likely to need a realtor for initial research. A key area in which value is added by Realtors is in advising clients on pricing. A realtor has access to all the historical sales data in the area, and based on that data, they suggest a range in which the property will likely sell (or the property is realistically worth). This is typically lower than the list price, and in most markets there is an expectation that there will be some negotiation (therefore some room is built into the list price to negotiate). We’ll be looking at the purchase side of the real estate transaction. Our objective was to see if by inputting all this data into a neural network, could we predict with some degree of accuracy what that sale price should be? 


# How to Run:

```
$ python3 realestate_regressor.py
```

The program will ask you to input parameters of an active listing and provide you with a selling price:
```
$ Please enter list price: 
```

# Output:
The program will export three files:
 - 2014-2019-cleaned.csv, which contains the entire cleaned dataset
 - output_prediction.csv, which shows the 100 test properties with actual sold prices, predicted prices, residuals and proportions 
 - prediction.png, which shows the visualization of actual sold prices and predicted prices