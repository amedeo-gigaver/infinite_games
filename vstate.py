import json
import sqlite3
import sys


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
        count(*) filter (where exported = '1') as exported_predictions,
        min(submitted) as oldest_prediction
        from predictions
    """
)
prediction_stat = c.fetchall()
print('Predictions: ', prediction_stat[0][0])
print('Predictions exported: ', event_stat[0][1])
print('Oldest prediction: ', event_stat[0][2])

if 'events' in (''.join(sys.argv)):
    c = cursor.execute(
        """
            select
            market_type, metadata, unique_event_id, description, registered_date
            from events where status = '2'
        """
    )
    result = c.fetchall()
    for market, metadata, event_id, title, reg_date in result:
        md = json.loads(metadata)
        sub_market = md.get('market_type', market)
        if 'events-id' in (''.join(sys.argv)):
            print(event_id)
        else:
            print(market, sub_market, event_id, title[:40], reg_date)

if 'events-predictions' in (''.join(sys.argv)):
    c = cursor.execute(
        """
            select
            market_type, metadata, unique_event_id, description, registered_date, count(*) as predictions
            from events predictions p inner join events e
                        on e.unique_event_id = p.unique_event_id where status = '2'
            group by unique_event_id, market_type, metadata, description, registered_date
        """
    )
    result = c.fetchall()
    for market, metadata, event_id, title, reg_date, preds in result:
        md = json.loads(metadata)
        sub_market = md.get('market_type', market)
        print(market, sub_market, event_id, title[:40], reg_date, preds)


cursor.close()
conn.close()
