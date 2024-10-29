import sqlite3


conn = sqlite3.connect('validator.db')

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
event_stat = c.fetchall()
print('In progress events: ', event_stat[0][0])
print('Exported events: ', event_stat[0][1])
print('Processed events: ', event_stat[0][2])
print('Settled events: ', event_stat[0][3])
print('Oldest event date: ', event_stat[0][4])
print('Recent register date: ', event_stat[0][5])
print('Events registered within a week: ', event_stat[0][6])
print('Events registered within a month: ', event_stat[0][7])

c = cursor.execute(
    """
        select
        count(*) as predictions,
        count(*) filter (where exported = '1') as exported_predictions
        from predictions
    """
)
prediction_stat = c.fetchall()
print('Predictions: ', prediction_stat[0][0])
print('Predictions exported: ', event_stat[0][1])

cursor.close()
conn.close()
