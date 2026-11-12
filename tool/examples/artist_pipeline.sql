WITH ArtistSales AS (
    SELECT a.ArtistId, ii.InvoiceId, i.Total
    FROM invoices i
    JOIN invoice_items ii ON i.InvoiceId = ii.InvoiceId
    JOIN tracks tr ON ii.TrackId = tr.TrackId
    JOIN albums al ON tr.AlbumId = al.AlbumId
    JOIN artists a ON al.ArtistId = a.ArtistId
),
TopArtist AS (
    SELECT ArtistId FROM ArtistSales
    GROUP BY ArtistId
    ORDER BY SUM(Total) DESC, (SELECT Name FROM artists WHERE ArtistId = ArtistSales.ArtistId) ASC
    LIMIT 1
),
BotArtist AS (
    SELECT ArtistId FROM ArtistSales
    GROUP BY ArtistId
    ORDER BY SUM(Total) ASC, (SELECT Name FROM artists WHERE ArtistId = ArtistSales.ArtistId) ASC
    LIMIT 1
),
TopAvg AS (
    SELECT SUM(Total) * CAST(COUNT(*) AS REAL) / COUNT(*) AS AvgAmt
    FROM ArtistSales JOIN invoices i ON ArtistSales.InvoiceId = i.InvoiceId
    WHERE ArtistSales.ArtistId = (SELECT ArtistId FROM TopArtist)
),
BotAvg AS (
    SELECT SUM(Total) * CAST(COUNT(*) AS REAL) / COUNT(*) AS AvgAmt
    FROM ArtistSales JOIN invoices i ON ArtistSales.InvoiceId = i.InvoiceId
    WHERE ArtistSales.ArtistId = (SELECT ArtistId FROM BotArtist)
)
SELECT ABS(TopAvg.AvgAmt - BotAvg.AvgAmt) FROM TopAvg JOIN BotAvg ON 1=1;
