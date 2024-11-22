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
print('Predictions exported: ', prediction_stat[0][1])
print('Oldest prediction: ', prediction_stat[0][2])


if '--predictions' in (''.join(sys.argv)):
    print('All events state with prediction numbers: ')
    c = cursor.execute(
        """
            select
            e.market_type, e.metadata,e.unique_event_id, e.description, e.registered_date, e.exported, e.processed, count(*) as preds
            from predictions p inner join events e
            on e.unique_event_id = p.unique_event_id
            group by e.unique_event_id, e.market_type, e.metadata, e.description, e.registered_date
        """
    )
    result = c.fetchall()
    for market, metadata, event_id, title, reg_date, exported, processed, preds in result:
        md = json.loads(metadata)
        sub_market = md.get('market_type', market)
        print(market, sub_market, event_id, reg_date, f'export:{exported}', f'processed:{processed}', 'predictions:', preds)

else:
    print('All events state: ')
    c = cursor.execute(
        """
            select
            status, market_type, metadata, unique_event_id, description, registered_date, exported, processed
            from events
            order by status
        """
    )
    result = c.fetchall()
    for status, market, metadata, event_id, title, reg_date, exported, processed in result:
        md = json.loads(metadata)
        sub_market = md.get('market_type', market)
        if 'events-id' in (''.join(sys.argv)):
            print(event_id)
        else:
            print(status, market, sub_market, event_id, reg_date, f'export:{exported}', f'processed:{processed}')


cursor.close()
conn.close()
