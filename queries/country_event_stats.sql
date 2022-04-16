with total_count as (
    select country,
        event_gender,
        count(1) as total_participants
    from {data}
    group by 1,2
    having count(1)>1000
)
select d.country,
    d.event,
    d.event_gender,
    avg(d.won_medal) as medal_rate,
    count(1) as participation_count,
    1.00 * count(1)/sum(count(1)) over (partition by d.country, d.event_gender) as participation_rate
from {data} d
join total_count tc
on d.country = tc.country
and d.event_gender = tc.event_gender
group by 1,2,3;
--having medal_rate > 0.00