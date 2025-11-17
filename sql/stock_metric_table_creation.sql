CREATE TABLE dbo.stock_metrics (
    id INT IDENTITY(1,1) PRIMARY KEY,
    ticker NVARCHAR(10),
    period_type NVARCHAR(20),
    period_label NVARCHAR(20),
    return_value FLOAT,
    created_at DATETIME DEFAULT GETDATE()
);


select * from stock_metrics;
SELECT TOP 100 * FROM dbo.stock_metrics ORDER BY created_at DESC;
