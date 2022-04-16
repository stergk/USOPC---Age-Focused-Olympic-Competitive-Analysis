select cast(person_id as int) person_id,
    class,
    competition_date,
    competition_city,
    competition_country,
    event_gender,
    case when event_gender = 'Men' then 1 else 0 end as gender_male,
    case when event_gender = 'Women' then 1 else 0 end as gender_female,
    event_name as event_and_gender,
    event_name_short as event,
    info_strada_sport_name as sport_name,
    medal,
    case when medal in ('G', 'S', 'B') then 1 else 0 end as won_medal,
    noc_name as country,
    case when noc_name = competition_country then 1 else 0 end as is_home_competition,
    person_age_years as age,
    rank
from {data}
where rank is not null
    and person_age_years is not null
    and person_id > 0