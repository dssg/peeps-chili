
-- Create San Jose entity_demos table
-- requires the features.acs_aggregation_imputed table to be in place
-- and contain all the entity/as_of_date pairs you'll need for the analysis

CREATE SCHEMA kit_bias_adj;

CREATE TABLE kit_bias_adj.entity_demos
AS
SELECT entity_id,
       as_of_date,
       CASE WHEN acs_entity_id_1y_hh_median_income_max < 55000 THEN 'under55k' ELSE 'over55k' END AS median_income,
       CASE WHEN acs_entity_id_1y_poverty_pct_max < 0.12 THEN 'low' ELSE 'high' END AS poverty_level,
       CASE WHEN acs_entity_id_1y_race_white_pct_max < 0.50 THEN 'nonwhite' ELSE 'white' END AS majority_white
FROM features.acs_aggregation_imputed
;
ALTER TABLE kit_bias_adj.entity_demos ADD PRIMARY KEY (entity_id, as_of_date);



-- features.acs_aggregation_imputed

-- acs_entity_id_1y_hh_median_income_max

-- acs_entity_id_1y_poverty_pct_max

-- acs_entity_id_1y_race_white_pct_max


-- -- median income: over/under 55k
-- -- poverty pct: over/under 12%
-- -- white pct: over/under 50%

-- WITH x AS (
--     SELECT entity_id, as_of_date,
--            acs_entity_id_1y_race_white_pct_max AS col
--     FROM features.acs_aggregation_imputed
-- )
-- SELECT COUNT(*), MIN(col), AVG(col), MAX(col),
--        SUM(CASE WHEN col=0 THEN 1 ELSE 0 END),
--        -- AVG(CASE WHEN col<55000 THEN 1 ELSE 0 END)
--        -- AVG(CASE WHEN col<0.12 THEN 1 ELSE 0 END)
--        AVG(CASE WHEN col<0.50 THEN 1 ELSE 0 END)
-- FROM x
-- ;




