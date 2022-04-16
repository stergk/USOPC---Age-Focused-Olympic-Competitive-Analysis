with home_medals as (
    select country,
        avg(won_medal) as medal_rate,
        count(1) as competition_count
    from {data} where is_home_competition = 1
    group by 1
    having medal_rate > 0.000
),
away_medals as (
    select country,
        avg(won_medal) as medal_rate,
        count(1) as competition_count
    from {data} where is_home_competition = 0
    group by 1
    having medal_rate > 0.000
)

select h.country, 
    h.medal_rate as home_medal_rate, 
    h.competition_count as home_competition_count,
    a.medal_rate as away_medal_rate,
    a.competition_count as away_competition_count
from home_medals h
join away_medals a
on h.country = a.country
