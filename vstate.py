import sqlite3


conn = sqlite3.connect('validator.db')

cursor = conn.cursor()
c = cursor.execute(
    """
        select event_id, market_type, description, starts, resolve_date, outcome,registered_date, local_updated_at,status, metadata, processed, exported
        from events
    """
)
result = c.fetchall()
if result:
    for r in result:
        print(r[0])
        print(f'{r[1]} - {r[2][:10]} -{r[3]} - {r[4]} - {r[5]} - {r[6]} - {r[7]} - {r[8]} - {r[10]} - {r[11]}' )

cursor.close()
conn.close()
