with sub as (
    select distinct person_id,
        country,
        gender_male,
        event,
        is_home_competition,
        rank,
        age,
        won_medal,
        competition_date,
        dense_rank() over (
            partition by person_id, event, gender_male
                order by competition_date desc
        ) as ranking
    from {data}
)

select f.person_id,
    f.event,
    f.gender_male,
    f.country,
    ft.competition_date as ft_competition_date,
    ft.rank as ft_rank,
    ft.age as ft_age,
    ft.won_medal as ft_won_medal,
    t.competition_date as t_competition_date,
    t.rank as t_rank,
    t.age as t_age,
    t.won_medal as t_won_medal,
    s.competition_date as s_competition_date,
    s.rank as s_rank,
    s.age as s_age,
    s.won_medal as s_won_medal,
    f.competition_date as f_competition_date,
    f.rank as f_rank,
    f.age as f_age,
    f.won_medal as f_won_medal,
    f.is_home_competition as f_is_home_competition
from sub f
join sub s
on f.person_id = s.person_id
and f.event = s.event
and f.gender_male = s.gender_male
and f.ranking = s.ranking - 1
join sub t
on f.person_id = t.person_id
and f.event = t.event
and f.gender_male = t.gender_male
and f.ranking = t.ranking - 2
join sub ft
on f.person_id = ft.person_id
and f.event = ft.event
and f.gender_male = ft.gender_male
and f.ranking = ft.ranking - 3;
