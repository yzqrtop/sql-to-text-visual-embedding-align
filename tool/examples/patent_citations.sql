SELECT p.patent_id, p.title
FROM patents p
WHERE p.patent_id IN (
    SELECT cited.patent_id
    FROM citations cited
    GROUP BY cited.patent_id
    HAVING COUNT(DISTINCT cited.citation_id) > 5
)
AND p.patent_id NOT IN (
    SELECT apps.patent_id
    FROM citations apps
    WHERE apps.citation_id IS NOT NULL
);
