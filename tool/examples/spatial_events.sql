SELECT
  region_id,
  ST_GEOGPOINT(longitude, latitude) AS location,
  COUNT(DISTINCT event_id) AS event_count
FROM
  `project.dataset.user_events` e
JOIN
  `project.dataset.user_regions` r
ON
  e.user_id = r.user_id
  AND e.event_time BETWEEN TIMESTAMP('2024-01-01') AND TIMESTAMP('2024-12-31')
GROUP BY
  region_id, location
ORDER BY
  event_count DESC;
