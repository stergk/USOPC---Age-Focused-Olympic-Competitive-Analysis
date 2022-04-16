select distinct person_id,
    event,
    gender_male,
    country,
    competition_date as f_competition_date,
    rank as f_rank,
    age as f_age,
    won_medal as f_won_medal,
    is_home_competition as f_is_home_competition
from {data}