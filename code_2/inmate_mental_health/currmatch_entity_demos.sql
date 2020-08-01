

-- NOTES
-- One-off script used to create an entity_demos table for the current matchdatetime
-- that can be used for training new models. Can't be used too well directly off the
-- shelf (note the hard-coded experiment hash, dates, etc) but including here since 
-- it may prove useful as a starting point if we wanted to build from it in the future

-- The resulting table here is referenced in the race_decoupled_test_20190923.yaml
-- which in turn needs to be modified to run models for each subgroup (BOTH in the
-- cohort logic and cohort name)


CREATE TABLE kit_bias_class_test.currmatch_entity_demos
AS
SELECT DISTINCT ON (entity_id, matchdatetime, as_of_date)
    l2.entity_id, l2.matchdatetime, ed.as_of_date, ed.race, ed.race_3way, ed.sex, ed.age_bucket
FROM kit_bias_class_test.entity_demos ed
JOIN entities.entity_source_match_lookup l1 ON ed.entity_id = l1.entity_id AND ed.matchdatetime = l1.matchdatetime
JOIN entities.entity_source_match_lookup l2 ON l1.source = l2.source AND l1.sourceid = l2.sourceid
WHERE l2.matchdatetime = (SELECT MAX(matchdatetime) FROM entities.entity_source_match_lookup)
ORDER BY entity_id, matchdatetime, as_of_date, RANDOM()
;

ALTER TABLE kit_bias_class_test.currmatch_entity_demos ADD PRIMARY KEY (entity_id, as_of_date);
CREATE INDEX ON kit_bias_class_test.currmatch_entity_demos(as_of_date);


-- Need as_of_dates back to 2013-04-01 in order to train new models with prediction dates starting 2014-04-01...


SELECT train_end_time, MAX(model_id) AS model_id
FROM model_metadata.experiment_models em
JOIN model_metadata.models m USING(model_hash)
WHERE em.experiment_hash = '09b3bcab5a6e1eb1c712571f6a5abb75'
    AND train_end_time IN ('2013-04-01'::DATE, '2013-08-01'::DATE, '2013-12-01'::DATE)
GROUP BY 1
;



CREATE LOCAL TEMPORARY TABLE tmp_entity_demo_2013
    ON COMMIT PRESERVE ROWS
    AS
WITH ref_models_pre AS (
  -- choose a random model for each train_end_time to pull out the cohort via
  -- the predictions table
    SELECT train_end_time, MAX(model_id) AS model_id
    FROM model_metadata.experiment_models em
    JOIN model_metadata.models m USING(model_hash)
    WHERE em.experiment_hash = '09b3bcab5a6e1eb1c712571f6a5abb75'
        AND train_end_time IN ('2013-04-01'::DATE, '2013-08-01'::DATE, '2013-12-01'::DATE)
    GROUP BY 1
)
, ref_models AS (
    SELECT rmp.model_id, rmp.train_end_time,
           ((mg.model_config->'matchdatetime')::VARCHAR)::TIMESTAMP AS matchdatetime
    FROM ref_models_pre rmp
    JOIN model_metadata.models m USING(model_id)
    JOIN model_metadata.model_groups mg USING(model_group_id)
)
, entity_lkup AS (
  SELECT DISTINCT
         p.entity_id AS old_entity_id,
         l1.matchdatetime AS old_matchdatetime,
         l2.entity_id AS curr_entity_id,
         p.as_of_date
  FROM test_results.predictions p
  JOIN ref_models rm USING(model_id)
  JOIN entities.entity_source_match_lookup l1 ON p.entity_id = l1.entity_id AND l1.matchdatetime = rm.matchdatetime
  JOIN entities.entity_source_match_lookup l2 ON l1.source = l2.source AND l1.sourceid = l2.sourceid
  WHERE l2.matchdatetime = (SELECT MAX(matchdatetime) FROM entities.entity_source_match_lookup)
)
, hispanic_flags as (
  -- unlike other features, take max() for hispanic flag since data tends to be more potentially
  -- prone to false negatives depending on how it is collected
  SELECT p.old_entity_id, p.as_of_date,
         MAX(CASE WHEN TRIM(ethnic_101) = 'Y' THEN 1 ELSE 0 END) AS hispanic_flag
  FROM entity_lkup p
  JOIN records.jims_booking_records b ON p.curr_entity_id = b.entity_id
  JOIN raw.jocojimsinmatedata r
    ON b.booking_no = encode(r.hash_booking_no_0, 'hex')
  -- hispanic ethnicity data only seems available starting in 2007
  WHERE b.knowledge_date >= '2007-01-01'::DATE
        AND b.knowledge_date <= p.as_of_date
  GROUP BY 1,2
)
, event_demo_info as (
  select p.old_entity_id, p.as_of_date,
         e.entity_info ->> 'race' AS race,
         e.entity_info ->> 'sex' as sex,
         extract(year FROM age(p.as_of_date, (e.entity_info ->> 'date_of_birth')::date)) AS age,
         e.knowledge_date
  from semantic.events e
  join entity_lkup p
  on e.entity_id = p.curr_entity_id and knowledge_date <= p.as_of_date
)
, event_demo_aggs as (
  select old_entity_id,
         as_of_date,
         race,
         sex,
         case
            when age < 18 then 'a under 18'
            when age between 18 and 30 then 'b 18 to 30'
            when age between 31 and 45 then 'c 31 to 45'
            when age between 46 and 59 then 'd 46 to 59'
            when age >= 60 then 'e 60 plus'
            else NULL
         end AS age_bucket,
         COUNT(*) AS count_by_demo,
         MAX(knowledge_date) AS max_knowledge_date
  from event_demo_info
  group by grouping sets ((old_entity_id, as_of_date, race), (old_entity_id, as_of_date, sex), (old_entity_id, as_of_date, age_bucket))
)
, race_count as (select distinct on (old_entity_id, as_of_date)
        old_entity_id,
        as_of_date,
        race,
        count_by_demo,
        max_knowledge_date
    FROM event_demo_aggs
    WHERE race IS NOT NULL
    order by
      old_entity_id,
      as_of_date,
      count_by_demo desc,
      max_knowledge_date desc,
      Random()
)
, age_count as (select distinct on (old_entity_id, as_of_date)
        old_entity_id,
        as_of_date,
        age_bucket,
        count_by_demo,
        max_knowledge_date
    FROM event_demo_aggs
    WHERE age_bucket IS NOT NULL
    order by
      old_entity_id,
      as_of_date,
      count_by_demo desc,
      max_knowledge_date desc,
      Random()
)
, sex_count as (select distinct on (old_entity_id, as_of_date)
        old_entity_id,
        as_of_date,
        sex,
        count_by_demo,
        max_knowledge_date
    FROM event_demo_aggs
    WHERE sex IS NOT NULL
    order by
      old_entity_id,
      as_of_date,
      count_by_demo desc,
      max_knowledge_date desc,
      Random()
)
select p.old_entity_id AS entity_id,
       p.old_matchdatetime AS matchdatetime,
       p.as_of_date,
       -- layer hispanic ethnicity flag over race (in practice, only a handful of non-white race in JIMS
       -- data is marked hispanic, so we let it take precedence)
       CASE WHEN hispanic_flag = 1 THEN 'H' ELSE rc.race END AS race,
       CASE WHEN hispanic_flag = 1 THEN 'H' WHEN rc.race IN ('B', 'H', 'W') THEN rc.race ELSE 'W' END AS race_3way,
       sex,
       age_bucket
from (SELECT DISTINCT old_entity_id, old_matchdatetime, as_of_date FROM entity_lkup) p
left join hispanic_flags
using(old_entity_id, as_of_date)
left join race_count rc
using(old_entity_id, as_of_date)
left join age_count
using(old_entity_id, as_of_date)
left join sex_count
using(old_entity_id, as_of_date)
-- TODO: Far better would be to alter the bias preprocessing downstream to
-- not calculate metrics for the extremely small groups. This is a temporary
-- patch to avoid div/0 errors, but it affects all demographic analyses
-- rather than just age
where age_bucket != 'a under 18'
;



INSERT INTO kit_bias_class_test.currmatch_entity_demos
SELECT DISTINCT ON (entity_id, matchdatetime, as_of_date)
    l2.entity_id, l2.matchdatetime, ed.as_of_date, ed.race, ed.race_3way, ed.sex, ed.age_bucket
FROM tmp_entity_demo_2013 ed
JOIN entities.entity_source_match_lookup l1 ON ed.entity_id = l1.entity_id AND ed.matchdatetime = l1.matchdatetime
JOIN entities.entity_source_match_lookup l2 ON l1.source = l2.source AND l1.sourceid = l2.sourceid
WHERE l2.matchdatetime = (SELECT MAX(matchdatetime) FROM entities.entity_source_match_lookup)
ORDER BY entity_id, matchdatetime, as_of_date, RANDOM()
;


