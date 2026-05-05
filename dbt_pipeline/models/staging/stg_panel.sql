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
        treated                                  as is_treated,
        post                                     as is_post_covid,
        covid_active                             as is_treated_post,
        vessel_density,
        dvm_amplitude,
        sst,
        log_chla,
        bathymetry,
        season_sin,
        season_cos,
        season_q,
        case
            when bathymetry < 500  then 'shallow'
            when bathymetry < 1500 then 'mid'
            else 'deep'
        end                                      as depth_zone,
        case when year < 2020 then 1 else 0 end  as is_pre_period_only
    from source
)

select * from staged