import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
import streamlit as st
plt.style.use('seaborn-poster')

st.title('Coronavirus (COVID-19) Visualization & Prediction')

st.markdown("""
Coronavirus is a family of viruses that are named after their spiky crown. The novel coronavirus, also known as SARS-CoV-2, is a contagious respiratory virus that first reported in Wuhan, China. On 2/11/2020, the World Health Organization designated the name COVID-19 for the disease caused by the novel coronavirus. This notebook aims at exploring COVID-19 through data analysis and projections.

Coronavirus Case Data is provided by Johns Hopkins University
Learn more from the World Health Organization
Learn more from the Centers for Disease Control and Prevention
Check out map visualizations from JHU CCSE Dashboard
Source code is also on my Github

""")

# Import confirmed table

#st.subheader('Confirmed Data')


@st.cache
def confirmed():
    confirmed_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

    return confirmed_df


confirmed_df = confirmed()

# st.dataframe(confirmed_df)


#st.subheader('Deaths Data')


@st.cache
def deaths():
    deaths_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    return deaths_df


deaths_df = deaths()

# st.dataframe(deaths_df)

#st.subheader('Latest Data')


@st.cache
def latest():
    latest_data = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/09-09-2022.csv')

    return latest_data


latest_data = latest()

# st.dataframe(latest_data)


# st.subheader('US_medical_data')


@st.cache
def US_medical():
    us_medical_data = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/09-09-2022.csv')

    return us_medical_data


us_medical_data = US_medical()

unique_countries = list(latest_data['Country_Region'].unique())
# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Select the country here : ",
    unique_countries
)


# st.dataframe(us_medical_data)

confirmed_cols = confirmed_df.keys()
deaths_cols = deaths_df.keys()

# Get all the dates for the ongoing coronavirus pandemic
confirmed = confirmed_df.loc[:, confirmed_cols[4]:]
deaths = deaths_df.loc[:, deaths_cols[4]:]


num_dates = len(confirmed.keys())
ck = confirmed.keys()
dk = deaths.keys()

world_cases = []
total_deaths = []
mortality_rate = []


for i in range(num_dates):
    confirmed_sum = confirmed[ck[i]].sum()
    death_sum = deaths[dk[i]].sum()

    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)

    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)


# Getting daily increases and moving averages

def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d


def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i+window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average


# window size
window = 7

# confirmed cases
world_daily_increase = daily_increase(world_cases)
world_confirmed_avg = moving_average(world_cases, window)
world_daily_increase_avg = moving_average(world_daily_increase, window)

# deaths
world_daily_death = daily_increase(total_deaths)
world_death_avg = moving_average(total_deaths, window)
world_daily_death_avg = moving_average(world_daily_death, window)


days_since_1_22 = np.array([i for i in range(len(ck))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)

# Future forcasting
days_in_future = 10
future_forcast = np.array(
    [i for i in range(len(ck)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

# Convert integer into datetime for better visualization
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append(
        (start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

# We are using data from 8/1/2022 to present for the prediction model
# slightly modify the data to fit the model better (regression models cannot pick the pattern), we are using data from 8/1/22 and onwards for the prediction modeling
days_to_skip = 922
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(
    days_since_1_22[days_to_skip:], world_cases[days_to_skip:], test_size=0.07, shuffle=False)


# st.write('Model for predicting # of confirmed cases. I am using support vector machine, bayesian ridge , and linear regression in this example. We will show the results in the later section.')

# # use this to find the optimal parameters for SVR
# c = [0.01, 0.1, 1]
# gamma = [0.01, 0.1, 1]
# epsilon = [0.01, 0.1, 1]
# shrinking = [True, False]

# svm_grid = {'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

# svm = SVR(kernel='poly', degree=3)
# svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)
# svm_search.fit(X_train_confirmed, y_train_confirmed)


# svm_search.best_params_

col1, col2 = st.columns(2)

c1 = st.container()
c2 = st.container()
c3 = st.container()

# svm_confirmed = svm_search.best_estimator_
svm_confirmed = SVR(shrinking=True, kernel='poly',
                    gamma=0.01, epsilon=1, degree=3, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)

# check against testing data
svm_test_pred = svm_confirmed.predict(X_test_confirmed)


#st.write('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
#st.write('MSE:', mean_squared_error(svm_test_pred, y_test_confirmed))

fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.plot(y_test_confirmed)
ax.plot(svm_test_pred)
ax.legend(['Test Data', 'SVM Predictions'])
# st.pyplot(fig)

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=3)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=3)
bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(
    X_train_confirmed)
bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(
    X_test_confirmed)
bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)

#st.subheader('Polynomial regression')
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
#st.write('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
#st.write('MSE:', mean_squared_error(test_linear_pred, y_test_confirmed))


#st.subheader('Linear Model Coefficient')
# linear_model.coef_

fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.plot(y_test_confirmed)
ax.plot(test_linear_pred)
ax.legend(['Test Data', 'Polynomial Regression Predictions'])
# st.pyplot(fig)

# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
                 'normalize': normalize}

bayesian = BayesianRidge(fit_intercept=False)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error',
                                     cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)


# st.write(bayesian_search.best_params_)


bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)

#st.write('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
#st.write('MSE:', mean_squared_error(test_bayesian_pred, y_test_confirmed))

fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.plot(y_test_confirmed)
ax.plot(test_bayesian_pred)
ax.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
# st.pyplot(fig)

#st.write('Worldwide Overview Graphing the number of confirmed cases, deaths, and mortality rate (CFR). This gives us a big picture of the ongoing pandemic.')

# helper method for flattening the data, so it can be displayed on a bar graph


def flatten(arr):
    a = []
    arr = arr.tolist()
    for i in arr:
        a.append(i[0])
    return a


graph1 = [world_cases, world_confirmed_avg, total_deaths, world_death_avg]
#graphs = list([world_cases, total_deaths])

# world_graphing1 = st.sidebar.selectbox(
#"Select the graph here : ",

#{'world': [world_cases]}
# )


adjusted_dates = adjusted_dates.reshape(1, -1)[0]
with col1:
    plt.figure(figsize=(16, 8))
    plt.plot(adjusted_dates, world_cases)
    plt.plot(adjusted_dates, world_confirmed_avg,
             linestyle='dashed', color='orange')
    plt.title('# of Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Worldwide Coronavirus Cases',
                'Moving Average {} Days'.format(window)], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    st.pyplot(plt)

with col2:
    plt.figure(figsize=(16, 8))
    plt.plot(adjusted_dates, total_deaths)
    plt.plot(adjusted_dates, world_death_avg,
             linestyle='dashed', color='orange')
    plt.title('# of Coronavirus Deaths Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Worldwide Coronavirus Deaths',
                'Moving Average {} Days'.format(window)], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    st.pyplot(plt)

with col1:
    plt.figure(figsize=(16, 10))
    plt.bar(adjusted_dates, world_daily_increase)
    plt.plot(adjusted_dates, world_daily_increase_avg,
             color='orange', linestyle='dashed')
    plt.title('World Daily Increases in Confirmed Cases', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Moving Average {} Days'.format(window),
                'World Daily Increase in COVID-19 Cases'], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    st.pyplot(plt)

with col2:
    plt.figure(figsize=(16, 10))
    plt.bar(adjusted_dates, world_daily_death)
    plt.plot(adjusted_dates, world_daily_death_avg,
             color='orange', linestyle='dashed')
    plt.title('World Daily Increases in Confirmed Deaths', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Moving Average {} Days'.format(window),
                'World Daily Increase in COVID-19 Deaths'], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    st.pyplot(plt)

with col1:
    plt.figure(figsize=(16, 10))
    plt.plot(adjusted_dates, np.log10(world_cases))
    plt.title('Log of # of Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    st.pyplot(plt)

with col2:
    plt.figure(figsize=(16, 10))
    plt.plot(adjusted_dates, np.log10(total_deaths))
    plt.title('Log of # of Coronavirus Deaths Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    st.pyplot(plt)

#st.subheader('Country Specific Graphs')
#st.write("Unlike the previous section, we are taking a look at specific countries. This allows us to examine covid at a more localized level. Feel free to change/edit this list to visulize the countries of your choice.")


def country_plot(x, y1, y2, y3, country):
    # window is set as 14 in in the beginning of the notebook
    confirmed_avg = moving_average(y1, window)
    confirmed_increase_avg = moving_average(y2, window)
    death_increase_avg = moving_average(y3, window)
    SIZE = (12, 8)
    with st.expander(f'Country Specific Graphs, now viewing cases for {add_selectbox}'):

        plt.figure(figsize=SIZE)
        plt.plot(x, y1)
        plt.plot(x, confirmed_avg, color='red', linestyle='dashed')
        plt.legend(['{} Confirmed Cases'.format(country),
                    'Moving Average {} Days'.format(window)], prop={'size': 20})
        plt.title('{} Confirmed Cases'.format(country), size=30)
        plt.xlabel('Days Since 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        st.pyplot(plt)

        plt.figure(figsize=SIZE)
        plt.bar(x, y2)
        plt.plot(x, confirmed_increase_avg,
                 color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(
            window), '{} Daily Increase in Confirmed Cases'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Confirmed Cases'.format(
            country), size=30)
        plt.xlabel('Days Since 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        st.pyplot(plt)

        plt.figure(figsize=SIZE)
        plt.bar(x, y3)
        plt.plot(x, death_increase_avg, color='red', linestyle='dashed')
        plt.legend(['Moving Average {} Days'.format(
            window), '{} Daily Increase in Confirmed Deaths'.format(country)], prop={'size': 20})
        plt.title('{} Daily Increases in Deaths'.format(country), size=30)
        plt.xlabel('Days Since 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        st.pyplot(plt)


# helper function for getting a country's total covid cases and deaths


def get_country_info(country_name):
    country_cases = []
    country_deaths = []

    for i in range(num_dates):
        country_cases.append(
            confirmed_df[confirmed_df['Country/Region'] == country_name][ck[i]].sum())
        country_deaths.append(
            deaths_df[deaths_df['Country/Region'] == country_name][dk[i]].sum())
    return (country_cases, country_deaths)


def country_visualizations(country_name):
    country_info = get_country_info(country_name)
    country_cases = country_info[0]
    country_deaths = country_info[1]

    country_daily_increase = daily_increase(country_cases)
    country_daily_death = daily_increase(country_deaths)

    country_plot(adjusted_dates, country_cases,
                 country_daily_increase, country_daily_death, country_name)


# top 10 total covid cases as of 7/13/2022
countries = ['US', 'India', 'Brazil', 'France', 'Germany', 'United Kingdom', 'Italy', 'Korea, South',
             'Russia', 'Turkey']

# for country in countries:
country_visualizations(add_selectbox)


# Country Comparison
# removed redundant code

# compare_countries = ['India', 'US', 'Brazil',
# 'Russia', 'United Kingdom', 'France']

# compare_countries = st.sidebar.multiselect(
#'Select the countries to compare :',
# unique_countries)

graph_name = ['Coronavirus Confirmed Cases', 'Coronavirus Confirmed Deaths']

with st.expander('Compare coronavirus cases for different countries'):
    st.subheader("Coronavirus cases for different countries")
    compare_countries = st.multiselect(
        'Select the countries to compare :',
        unique_countries)

    for num in range(2):
        plt.figure(figsize=(12, 8))
        for country in compare_countries:
            plt.plot(get_country_info(country)[num])
        plt.legend(compare_countries, prop={'size': 20})
        plt.xlabel('Days since 1/22/2020', size=30)
        plt.ylabel('# of Cases', size=30)
        plt.title(graph_name[num], size=30)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        st.pyplot(plt)


def plot_predictions(x, y, pred, algo_name, color):
    plt.figure(figsize=(12, 8))
    plt.plot(x, y)
    plt.plot(future_forcast, pred, linestyle='dashed', color=color)
    plt.title('Worldwide Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Confirmed Cases', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    st.pyplot(plt)


with st.expander('Predictions for confirmed coronavirus cases worldwide'):
    st.markdown("""These three models predict future covid cases on a global level. 
    These are constructed to use the latest window of data to predict the current trend.
    The prediction models include : Support Vector Machine Polynomial Regression Bayesian Ridge Regression""")

    preds = {
        'SVM Predictions': svm_pred,
        'Polynomial Regression Predictions': linear_pred,
        'Bayesian Ridge Regression Predictions': bayesian_pred,
    }

    model_predict = st.sidebar.selectbox(
        "Select the model to predict : ", list(preds.keys()))

# ['SVM Predictions', 'Polynomial Regression Predictions',
    # 'Bayesian Ridge Regression Predictions']
    if model_predict == 'SVM Predictions':
        plot_predictions(adjusted_dates, world_cases,
                         preds[model_predict], 'SVM Predictions', 'purple')
    elif model_predict == 'Polynomial Regression Predictions':
        plot_predictions(adjusted_dates, world_cases, preds[model_predict],
                         'Polynomial Regression Predictions', 'orange')
    elif model_predict == 'Bayesian Ridge Regression Predictions':
        plot_predictions(adjusted_dates, world_cases, preds[model_predict],
                         'Bayesian Ridge Regression Predictions', 'green')
    else:
        pass

with st.expander('Future Predictions'):

    if model_predict == 'SVM Predictions':
        st.subheader(f'Future predictions using {model_predict}')
        svm_df = pd.DataFrame(
            {'Date': future_forcast_dates[-10:], 'SVM Predicted # of Confirmed Cases Worldwide': np.round(preds[model_predict][-10:])})

        st.dataframe(svm_df.style.background_gradient(cmap='Reds'))

    elif model_predict == 'Polynomial Regression Predictions':
        st.subheader(f'Future predictions using {model_predict}')
        linear_pred = linear_pred.reshape(1, -1)[0]
        linear_df = pd.DataFrame(
            {'Date': future_forcast_dates[-10:], 'Polynomial Predicted # of Confirmed Cases Worldwide': np.round(preds[model_predict][-10:])})
        st.dataframe(linear_df.style.background_gradient(cmap='Reds'))

    elif model_predict == 'Bayesian Ridge Regression Predictions':

        st.subheader(f'Future predictions using {model_predict}')
        bayesian_df = pd.DataFrame(
            {'Date': future_forcast_dates[-10:], 'Bayesian Ridge Predicted # of Confirmed Cases Worldwide': np.round(preds[model_predict][-10:])})
        st.dataframe(bayesian_df.style.background_gradient(cmap='Reds'))

    else:
        pass

with st.expander('Worldwide Mortality Rate of Coronavirus Over Time'):
    mean_mortality_rate = np.mean(mortality_rate)
    plt.figure(figsize=(16, 10))
    plt.plot(adjusted_dates, mortality_rate, color='orange')
    plt.axhline(y=mean_mortality_rate, linestyle='--', color='black')
#plt.title('Worldwide Mortality Rate of Coronavirus Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Case Mortality Rate', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    st.pyplot(plt)


unique_countries = list(latest_data['Country_Region'].unique())

country_confirmed_cases = []
country_death_cases = []
country_active_cases = []
country_incidence_rate = []
country_mortality_rate = []

no_cases = []
for i in unique_countries:
    cases = latest_data[latest_data['Country_Region'] == i]['Confirmed'].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)

for i in no_cases:
    unique_countries.remove(i)


# sort countries by the number of confirmed cases
unique_countries = [k for k, v in sorted(zip(
    unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_data[latest_data['Country_Region']
                                             == unique_countries[i]]['Confirmed'].sum()
    country_death_cases.append(
        latest_data[latest_data['Country_Region'] == unique_countries[i]]['Deaths'].sum())
    country_incidence_rate.append(
        latest_data[latest_data['Country_Region'] == unique_countries[i]]['Incident_Rate'].sum())
    country_mortality_rate.append(
        country_death_cases[i]/country_confirmed_cases[i])

#st.subheader('Data table')
with st.expander('Worldwide Datatable'):
    st.markdown("""This shows covid data for several countries. 
    The table includes the number of confirmed cases, deaths, incidence rate, and mortality rate.""")

    country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': [format(int(i), ',d') for i in country_confirmed_cases],
                               'Number of Deaths': [format(int(i), ',d') for i in country_death_cases],
                               'Incidence Rate': country_incidence_rate,
                               'Mortality Rate': country_mortality_rate})
# number of cases per country/region

    st.dataframe(country_df.style.background_gradient(cmap='Oranges'))

unique_provinces = list(latest_data['Province_State'].unique())

province_confirmed_cases = []
province_country = []
province_death_cases = []
province_incidence_rate = []
province_mortality_rate = []

no_cases = []
for i in unique_provinces:
    cases = latest_data[latest_data['Province_State'] == i]['Confirmed'].sum()
    if cases > 0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)

# remove areas with no confirmed cases
for i in no_cases:
    unique_provinces.remove(i)

unique_provinces = [k for k, v in sorted(zip(
    unique_provinces, province_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_provinces)):
    province_confirmed_cases[i] = latest_data[latest_data['Province_State']
                                              == unique_provinces[i]]['Confirmed'].sum()
    province_country.append(
        latest_data[latest_data['Province_State'] == unique_provinces[i]]['Country_Region'].unique()[0])
    province_death_cases.append(
        latest_data[latest_data['Province_State'] == unique_provinces[i]]['Deaths'].sum())
#     province_recovery_cases.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Recovered'].sum())
#     province_active.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Active'].sum())
    province_incidence_rate.append(
        latest_data[latest_data['Province_State'] == unique_provinces[i]]['Incident_Rate'].sum())
    province_mortality_rate.append(
        province_death_cases[i]/province_confirmed_cases[i])

# number of cases per province/state/city top 100
province_limit = 100
province_df = pd.DataFrame({'Province/State Name': unique_provinces[:province_limit], 'Country': province_country[:province_limit], 'Number of Confirmed Cases': [format(int(i), ',d') for i in province_confirmed_cases[:province_limit]],
                            'Number of Deaths': [format(int(i), ',d') for i in province_death_cases[:province_limit]],
                            'Incidence Rate': province_incidence_rate[:province_limit], 'Mortality Rate': province_mortality_rate[:province_limit]})
# number of cases per country/region

province_df.style.background_gradient(cmap='Oranges')


# return the data table with province/state info for a given country
def country_table(country_name):
    states = list(latest_data[latest_data['Country_Region']
                  == country_name]['Province_State'].unique())
    state_confirmed_cases = []
    state_death_cases = []
    # state_recovery_cases = []
#     state_active = []
    state_incidence_rate = []
    state_mortality_rate = []

    no_cases = []
    for i in states:
        cases = latest_data[latest_data['Province_State']
                            == i]['Confirmed'].sum()
        if cases > 0:
            state_confirmed_cases.append(cases)
        else:
            no_cases.append(i)

    # remove areas with no confirmed cases
    for i in no_cases:
        states.remove(i)

    states = [k for k, v in sorted(
        zip(states, state_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
    for i in range(len(states)):
        state_confirmed_cases[i] = latest_data[latest_data['Province_State']
                                               == states[i]]['Confirmed'].sum()
        state_death_cases.append(
            latest_data[latest_data['Province_State'] == states[i]]['Deaths'].sum())
    #     state_recovery_cases.append(latest_data[latest_data['Province_State']==states[i]]['Recovered'].sum())
#         state_active.append(latest_data[latest_data['Province_State']==states[i]]['Active'].sum())
        state_incidence_rate.append(
            latest_data[latest_data['Province_State'] == states[i]]['Incident_Rate'].sum())
        state_mortality_rate.append(
            state_death_cases[i]/state_confirmed_cases[i])

    state_df = pd.DataFrame({'State Name': states, 'Number of Confirmed Cases': [format(int(i), ',d') for i in state_confirmed_cases],
                             'Number of Deaths': [format(int(i), ',d') for i in state_death_cases],
                             'Incidence Rate': state_incidence_rate, 'Mortality Rate': state_mortality_rate})
    # number of cases per country/region
    return state_df


#st.subheader('Data table for selected country')
with st.expander(f'Data table for selected country : {add_selectbox}'):
    india_table = country_table(add_selectbox)
    st.dataframe(india_table.style.background_gradient(cmap='Oranges'))

#st.subheader("Bar Chart Visualizations for COVID-19")

total_world_cases = np.sum(country_confirmed_cases)
us_confirmed = latest_data[latest_data['Country_Region']
                           == add_selectbox]['Confirmed'].sum()
outside_us_confirmed = total_world_cases - us_confirmed
with st.expander(f'Bar Chart Visualizations for COVID-19 : {add_selectbox}'):
    st.markdown(
        """This offers us some insights for how select countries/regions look on covid cases.""")
    plt.figure(figsize=(16, 9))
    plt.barh(add_selectbox, us_confirmed)
    plt.barh(f'Outside {add_selectbox}', outside_us_confirmed)
    plt.title('# of Total Coronavirus Confirmed Cases', size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    st.pyplot(plt)

    plt.figure(figsize=(16, 9))
    plt.barh(add_selectbox, us_confirmed/total_world_cases)
    plt.barh(f'Outside {add_selectbox}',
             outside_us_confirmed/total_world_cases)
    plt.title('# of Coronavirus Confirmed Cases Expressed in Percentage', size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
    st.pyplot(plt)


# Only show 10 countries with the most confirmed cases, the rest are grouped into the other category
visual_unique_countries = []
visual_confirmed_cases = []
others = np.sum(country_confirmed_cases[10:])

for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])

visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)

with st.expander('Bar Chart Visualizations for COVID-19  for other countries'):
    st.markdown(
        """This offers us some insights for how select countries/regions compare on covid cases.""")

    def plot_bar_graphs(x, y, title):
        plt.figure(figsize=(16, 12))
        plt.barh(x, y)
        plt.title(title, size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.show()
        st.pyplot(plt)

# good for a lot x values

    def plot_bar_graphs_tall(x, y, title):
        plt.figure(figsize=(19, 18))
        plt.barh(x, y)
        plt.title(title, size=25)
        plt.xticks(size=25)
        plt.yticks(size=25)
        plt.show()
        st.pyplot(plt)

    plot_bar_graphs(visual_unique_countries, visual_confirmed_cases,
                    '# of Covid-19 Confirmed Cases in Countries/Regions')

    log_country_confirmed_cases = [
        math.log10(i) for i in visual_confirmed_cases]
    plot_bar_graphs(visual_unique_countries, log_country_confirmed_cases,
                    'Common Log # of Coronavirus Confirmed Cases in Countries/Regions')

# Only show 10 provinces with the most confirmed cases, the rest are grouped into the other category
    visual_unique_provinces = []
    visual_confirmed_cases2 = []
    others = np.sum(province_confirmed_cases[10:])
    for i in range(len(province_confirmed_cases[:10])):
        visual_unique_provinces.append(unique_provinces[i])
        visual_confirmed_cases2.append(province_confirmed_cases[i])

    visual_unique_provinces.append('Others')
    visual_confirmed_cases2.append(others)

    plot_bar_graphs(visual_unique_provinces, visual_confirmed_cases2,
                    '# of Coronavirus Confirmed Cases in Provinces/States')

    log_province_confirmed_cases = [
        math.log10(i) for i in visual_confirmed_cases2]
    plot_bar_graphs(visual_unique_provinces, log_province_confirmed_cases,
                    'Log of # of Coronavirus Confirmed Cases in Provinces/States')
