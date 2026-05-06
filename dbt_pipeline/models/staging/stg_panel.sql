with source as (
    select * from raw_panel
),

staged as (
    select
        cell_id,
        week_index,
        date::date                               as observation_date,
        year,
        week,
        is_treated,
        is_post_covid,
        is_treated_post,
        vessel_density,
        dvm_amplitude,
        sst,
        log_chla,
        bathymetry,
        season_sin,
        season_cos,
        season_q,
        depth_zone::varchar                      as depth_zone,
        is_pre_period_only
    from source
)

select * from staged