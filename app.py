import pandas as pd
#data = pd.read_csv("~/Downloads/charging_station_data.csv")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# read csv file into dataframe
data = pd.read_csv("~/Downloads/charging_station_data.csv")

# # filter data for only JMD Square fast charging station
data = data[(data['Location'] == "Sector 15, Gurugram")]
#
# # calculate revenue by multiplying number of kW by cost per kW
data['revenue'] = data['kW'] * data['cost per kW']
print(data)
#
# # define features and target variable
# X = data[['day_of_week', 'time_of_day']]
# y = data['revenue']
#
# # one-hot encode day_of_week feature
# X = pd.get_dummies(X, columns=['day_of_week'])
#
# # split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # train linear regression model
# linear_reg = LinearRegression()
# linear_reg.fit(X_train, y_train)
#
# # make predictions on test data
# y_pred = linear_reg.predict(X_test)
#
# # calculate mean squared error of model
# mse = mean_squared_error(y_test, y_pred)
#
# print("Mean Squared Error of model: ", mse)
#
# # predict revenue for tomorrow
# tomorrow = {"day_of_week_Monday": 0, "day_of_week_Tuesday": 0, "day_of_week_Wednesday": 0, "day_of_week_Thursday": 0, "day_of_week_Friday": 0, "day_of_week_Saturday": 0, "day_of_week_Sunday": 1, "time_of_day": 15, "temperature": 25}
# tomorrow = pd.DataFrame(tomorrow, index=[0])
# predicted_revenue = linear_reg.predict(tomorrow)
# print("Predicted revenue for tomorrow: ", predicted_revenue[0])
