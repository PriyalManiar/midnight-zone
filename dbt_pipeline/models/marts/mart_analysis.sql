with staged as (
    select * from {{ ref('stg_panel') }}
),

weekly_group_means as (
    select
        week_index,
        observation_date,
        is_treated,
        avg(vessel_density) as mean_vessel_density,
        avg(dvm_amplitude)  as mean_dvm_amplitude,
        count(*)            as n_cells
    from staged
    group by week_index, observation_date, is_treated
),

final as (
    select
        s.*,
        w.mean_vessel_density  as group_week_mean_density,
        w.mean_dvm_amplitude   as group_week_mean_dvm,
        (s.sst       - 14.7)  / 4.3    as sst_std,
        (s.log_chla  - 0.48)  / 0.65   as log_chla_std,
        (s.bathymetry - 1462) / 1194   as bathymetry_std
    from staged s
    left join weekly_group_means w
        on s.week_index = w.week_index
        and s.is_treated = w.is_treated
)

select * from final