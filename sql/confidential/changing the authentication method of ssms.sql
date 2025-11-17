--enable sa if disabled
ALTER LOGIN [sa] ENABLE;

-- set or reset the password for sa
ALTER LOGIN [sa] WITH PASSWORD = N'StrongPassword123#';

--disabled password policies (for dev use only)
ALTER LOGIN [sa] WITH CHECK_POLICY = OFF, CHECK_EXPIRATION = OFF;

--verify status
SELECT name, is_disabled FROM sys.sql_logins WHERE name = 'sa';