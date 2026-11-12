SELECT T1.driverId
FROM lapTimes AS T1
INNER JOIN races AS T2 ON T1.raceId = T2.raceId
WHERE T2.name = 'French_Grand_Prix' AND T1.lap = 3
ORDER BY T1.time DESC
LIMIT 1;
