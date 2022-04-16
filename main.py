import numpy as np
import pandas as pd
from pandasql import sqldf
import itertools
from utils import read_query, process_data


def create_features(filepath):
    """
    Performs feature engineering and creates features and response variables according to specified logic.
    :param filepath: path of file to be processed.
    :return:
    """

    data = process_data(filepath)

    # Age one-hot_encoding
    age_one_hot_query = """
    select case when age < 12 then 1 else 0 end as age_less_12,
        {other_cases},
        case when age > 60 then 1 else 0 end as age_more_60
    from data
    """.format(other_cases=',\n'.join([f"""case when age = {x} then 1 else 0 end as age_{x}""" for x in range(12, 61)]))
    age_one_hot = sqldf(query=age_one_hot_query)

    # Class determination and one-hot-encoding
    unique_class = data['class'].unique().tolist()
    class_one_hot_query = """
    select {cases}
    from data
    """.format(
        cases=',\n'.join([f"""case when class = '{x}' then 1 else 0 end as class_{x.lower()}""" for x in unique_class])
    )
    class_one_hot = sqldf(class_one_hot_query)

    # Event name one-hot-encoding
    unique_events = data['event_name_short'].unique().tolist()
    event_one_hot_query = """
    select {cases}
    from data
    """.format(
        cases=',\n'.join(
            [f"""case when event_name_short = '{x}' then 1 else 0 end as event_{x.lower().replace(' ','_')
                .replace('/', '_')
                .replace('-','_')
            }""" for x in unique_events])
    )
    event_one_hot = sqldf(event_one_hot_query)

    # Polynomial features
    age_col = age_one_hot.columns.tolist()
    class_col = class_one_hot.columns.tolist()
    event_col = event_one_hot.columns.tolist()

    cols_list = [class_col, age_col, event_col]
    polynomial_df = sqldf("""select medal, rank, gender_male, gender_female from data""")

    combos = list(itertools.product(*cols_list))

    for cls, age, event in combos:
        new_col = class_one_hot[cls] * age_one_hot[age] * event_one_hot[event]
        polynomial_df['__'.join([cls, age, event])] = new_col

    polynomial_df.to_csv(filepath.split('.csv')[0] + '_processed_features.csv')

    return polynomial_df


if __name__ == '__main__':
    file_name = 'data/vw_EventResults_Podium_WithTeamSelect (infostrada)_Snowboard.csv'
    # processed = process_data(file_name)
    final = create_features(file_name)
