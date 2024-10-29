import sqlite3


conn = sqlite3.connect('test.db')

cursor = conn.cursor()
c = cursor.execute(
    """
        select
        count(*) filter (where status = '2') as pending,
        count(*) filter (where exported = '1') as exported,
        count(*) filter (where processed = '1') as processed,
        count(*) filter (where status = '3') as resolved,
        min(registered_date) as older_event,
        max(registered_date) as recently_registered,
        count(*) filter (where registered_date > date('now', '-1 weeks')) as last_week_events,
        count(*) filter (where registered_date > date('now', '-1 months')) as last_month_events
        from events
    """
)
result = c.fetchall()
cursor.close()
conn.close()
print('In progress events: ', result[0][0])
print('Exported events: ', result[0][1])
print('Processed events: ', result[0][2])
print('Settled events: ', result[0][3])
print('Oldest event date: ', result[0][4])
print('Recent register date: ', result[0][5])
print('Events registered within a week: ', result[0][6])
print('Events registered within a month: ', result[0][7])
