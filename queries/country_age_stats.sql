with total_count as (
    select country,
        event_gender,
        count(1) as total_participants
    from {data}
    group by 1,2
    having count(1)>1000
)
select d.country,
    d.event_gender,
    cast(d.age as integer) AS age,
    avg(d.won_medal) as medal_rate,
    count(1) as participation_count,
    1.00 * count(1)/sum(count(1)) over (partition by d.country, d.event_gender) as participation_rate
from {data} d
join total_count tc
on tc.country = d.country
and tc.event_gender = d.event_gender
group by 1,2,3;
--having avg(won_medal)>0.0
