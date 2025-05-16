mysql> CREATE TABLE Distance AS SELECT     CustomerId,     Lng,     Lat,     Zip,    ST_DISTANCE_SPHERE(POINT(Lng, Lat), POINT(-71.10253, 42.36224)) AS distance FROM Customer JOIN Zips ON PostalCode = Zip HAVING  distance < 100000;
--Creating a new table distance if the  zips are within 100km 
Query OK, 62939 rows affected (9.29 sec)
Records: 62939  Duplicates: 0  Warnings: 0

mysql> SHOW Tables; -- to view if the new table distance was created
+--------------------+
| Tables_in_shanmuga |
+--------------------+
| Customer           |
| Distance           |
| Line               |
| OrderDetail        |
| Orders             |
| Product            |
| Vendor             |
| Zips               |
+--------------------+
8 rows in set (0.00 sec)

mysql> SELECT 
    ->     d.Zip,
    ->     MIN(d.distance) AS MinDistance,  -- Get the minimum distance for each Zip
    ->     SUM(q.QuantityOrdered) AS TotalQuantityOrdered  -- Sum of QuantityOrdered for each Zip
    -> FROM 
    ->     Distance d  
    -> JOIN 
    ->     Orders o ON d.CustomerID = o.CustomerID 
    -> JOIN 
    ->     OrderDetail q ON o.OrderId = q.OrderId 
    -> GROUP BY 
    ->     d.Zip  -- Group by Zip
    -> ORDER BY 
    ->     TotalQuantityOrdered DESC,  -- Order by total QuantityOrdered in decreasing order
    ->     MinDistance ASC  -- Then order by minimum distance in increasing order
    -> LIMIT 3;

    +------+--------------------+----------------------+
| Zip  | MinDistance        | TotalQuantityOrdered |
+------+--------------------+----------------------+
| 2169 |  15295.03061303072 |                 6579 |
| 2155 |  6802.267177011108 |                 5769 |
| 2446 | 2616.6809670363505 |                 3611 |
+------+--------------------+----------------------+
3 rows in set (2.23 sec)