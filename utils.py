import pandas as pd
import numpy as np
from pandasql import sqldf
import os
from varname import nameof
from geopy.geocoders import Nominatim
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
RandomForestClassifier,
RandomForestRegressor,
GradientBoostingClassifier,
GradientBoostingRegressor
)
from sklearn.metrics import (
confusion_matrix,mean_absolute_error,classification_report, mean_squared_error
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight


def read_query(data, query):
    """
    Returns a dataframe with the requested query from the queries folder if the file exists
    or executes query directly otherwise
    :param data: data source to query from
    :param query: name of query file or query itself
    :return: dataframe with query results
    """

    assert '.sql' in query or ' ' in query, "Query file invalid. Please check"

    if query.endswith('.sql'):
        path_to_query = os.path.join(
            os.getcwd(), 'queries', query
        )
        file = open(path_to_query, 'r')
        query = file.read()
        query = query.format(data=nameof(data))

    df = sqldf(query=query)

    return df


def process_data(file, from_query=True) -> pd.DataFrame:
    """
    Processes the dataset and removes unnecessary rows and columns

    :param file: filepath for csv file
    :param from_query: will the output be derived from specified query
    :return: Pandas Dataframe
    """
    data = pd.read_csv(file, low_memory=False)
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]

    cities = data.competition_city.unique()
    for city in cities:
        if city is not np.nan:
            country_from_city(city)

    data['competition_country'] = data.competition_city.apply(lambda x: country_from_city(x) if x is not np.nan else None)
    data['competition_date'] = pd.to_datetime(data.competition_date)
    data['competition_date'] = data.competition_date.dt.strftime('%Y-%m-%d')

    sport = file.split('_')[-1].split('.')[0]
    query = 'data_processing.sql'

    if from_query:
        final = read_query(data, query)
        output_file = sport + '_processed.csv'
        output_path = os.path.join('data', output_file)
        final.to_csv(output_path, index=False)
    else:
        final = data
        output_file = sport + '_raw.csv'
        output_path = os.path.join('data', output_file)
        final.to_csv(output_path, index=False)

    return final


def country_from_city(city, log=False):

    """
    Deternmines country based on provided city
    :param city: city to find country (str)
    :return: country (str)
    """

    mappings_path = os.path.join(os.getcwd(), 'extras', 'city_country.json')
    with open(mappings_path, 'r') as f:
        j = f.read()
    mappings = json.loads(j)

    if city is not np.nan:
        if city in mappings.keys():
            if log:
                print(f'Looking up city {city}...')
            country = mappings[city]
            if log:
                print(f'{city} is in {country}')
            return country

        else:
            gl = Nominatim(user_agent='Some')
            if log:
                print(f'Looking up city {city}...')
            loc = gl.geocode(city, language='en')
            if loc:
                country = loc.address.split(', ')[-1]
                if country == 'Czechia':
                    country = 'Czech Republic'
                elif country == 'United Kingdom':
                    country = 'Great Britain'
                elif 'Korea' in country:
                    country = 'Korea'
            else:
                country = None
            mappings[city] = country
            if log:
                print(f'{city} is in {country}')

        with open(mappings_path, 'w') as fp:
            json.dump(mappings, fp)
    else:
        country = None

    return country


def create_heatmap(df, x, y, val, filtering=None, figsize=(20, 10), ret_df=False, save=False, fontsize=15):

    img_title = f'{x.capitalize()} by {y.capitalize()} - {val.replace("_"," ").capitalize()}\n'

    if filtering:
        for k, v in filtering.items():
            df = df[df[k] == v]
        img_title += ' - '.join([k.capitalize().replace('_', ' ') + '=' + v for k, v in filtering.items()]) + '\n'

    heat = df[[x, y, val]].pivot_table(index=y, columns=x).fillna(0).round(2)
    heat.columns = heat.columns.droplevel()

    if heat.shape[0] == 1:
        heat.T.plot.bar(figsize=figsize, grid='both', title=img_title)
        plt.title(img_title, fontsize=fontsize)
    else:
        plt.figure(figsize=figsize)
        plt.title(img_title, fontsize=fontsize)
        sns.heatmap(heat, annot=True, fmt='g', linewidths=.5)

    if save:
        plt.savefig('img.png')
    if ret_df:
        return heat


def create_time_series(data):
    new_data = read_query(data, "time_series_processing.sql")
    return new_data


def create_reg(data):
    new_data = read_query(data, "regular_processing.sql")
    return new_data


def create_features(data, event, response_var, with_ts=True):
    if with_ts:
        data = create_time_series(data)
    else:
        data = create_reg(data)
    winning_countries = sorted(data[data.f_won_medal == 1].country.unique())
    for c in winning_countries:
        data.loc[:, c] = data.country.apply(lambda x: 1 if x == c else 0)
    data.drop('country', axis=1, inplace=True)
    if with_ts:
        data = data[data.event == event].drop(
            ['person_id',
             'event',
             't_competition_date',
             's_competition_date',
             'f_competition_date',
             'ft_competition_date'
             ], axis=1
        )
    else:
        data = data[data.event == event].drop(
            ['person_id',
             'event',
             'f_competition_date'
             ], axis=1
        )
    X = data.drop(['f_won_medal','f_rank'], axis=1)
    y = data[response_var]
    return X, y


def classifier(X, y, model, param_grid=None, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    if 'forest' in model.lower():
        weights = (pd.value_counts(y) / pd.value_counts(y)[1]).astype(int)
        weights = {0: weights[1], 1: weights[0]}
        rfc = RandomForestClassifier(n_estimators=20, class_weight=weights, random_state=random_state)
        if not param_grid:
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [10, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [2, 4, 6]
            }
        gs_classifier = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        gs_classifier.fit(X_train, y_train)
    elif 'gradient' in model.lower():
        weights = compute_sample_weight('balanced', y_train)
        gbc = GradientBoostingClassifier(n_estimators=10, random_state=random_state)
        if not param_grid:
            param_grid = {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [10, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [2, 4, 6]
            }
        gs_classifier = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5)
        gs_classifier.fit(X_train, y_train, sample_weight=weights)
    best = gs_classifier.best_estimator_
    print(gs_classifier.best_params_)
    y_test_pred = best.predict(X_test)
    y_train_pred = best.predict(X_train)
    print("""##################### Training Set ######################\n""")
    print(classification_report(y_true=y_train, y_pred=y_train_pred))
    print(confusion_matrix(y_true=y_train, y_pred=y_train_pred))
    print("""####################### Test Set ########################\n""")
    print(classification_report(y_true=y_test, y_pred=y_test_pred))
    print(confusion_matrix(y_true=y_test, y_pred=y_test_pred))
    return best


def regressor(X, y, model, param_grid=None, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    if 'forest' in model.lower():
        rfr = RandomForestRegressor(n_estimators=20, random_state=random_state)
        if not param_grid:
            param_grid = {
                'max_depth': [10, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [2, 4, 6]
            }
        est = rfr
    elif 'gradient' in model.lower():
        gbr = GradientBoostingRegressor(n_estimators=10, random_state=random_state)
        if not param_grid:
            param_grid = {
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [10, 50, 100],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [2, 4, 6]
            }
        est = gbr
    gs_regressor = GridSearchCV(estimator=est, param_grid=param_grid, cv=5)
    gs_regressor.fit(X_train, y_train)
    best = gs_regressor.best_estimator_
    print(gs_regressor.best_params_)
    y_test_pred = best.predict(X_test)
    y_train_pred = best.predict(X_train)

    print("""##################### Training Set ######################\n""")
    print('MAU:', mean_absolute_error(y_true=y_train, y_pred=y_train_pred))
    print('MSE:', mean_squared_error(y_true=y_train, y_pred=y_train_pred))

    print("""####################### Test Set ########################\n""")
    print('MAU:', mean_absolute_error(y_true=y_test, y_pred=y_test_pred))
    print('MSE:', mean_squared_error(y_true=y_test, y_pred=y_test_pred))

    return best


def feature_importance(model, x_train, title, figsize=(10, 4)):
    fi = sorted(
        [
            (f, i) for f, i in zip(x_train, model.feature_importances_)
        ],
        key=lambda x: x[1],
        reverse=True
    )
    imps = {}
    for f, i in fi:
        if i > 0.01:
            imps[f] = i
    imps = pd.Series(imps).sort_values(ascending=False)
    imps.plot.bar(
        figsize=figsize,
        grid='both',
        title=title
    )
    plt.xticks(rotation=90)
    plt.title(title, fontsize=15)
    plt.show()


if __name__ == '__main__':
    data = process_data('data/vw_EventResults_Podium_WithTeamSelect (infostrada)_Snowboard.csv', from_query=False)
    df_country_event_stats = read_query(data, 'country_event_stats.sql')
