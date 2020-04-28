import datetime
from dateutil.relativedelta import relativedelta

import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
def get_past_date(str_days_ago, start_date = datetime.date.today()):
    splitted = str_days_ago.split()
    if splitted[0] == 'a':
        splitted[0] = 1
    if len(splitted) == 1 and splitted[0].lower() == 'today':
        return str(start_date.isoformat())
    elif len(splitted) == 1 and splitted[0].lower() == 'yesterday':
        date = start_date - relativedelta(days=1)
        return str(date.isoformat())
    elif len(splitted) == 1:
        np.nan
    elif splitted[1].lower() in ['hour', 'hours', 'hr', 'hrs', 'h']:
        date = start_date - relativedelta(hours=int(splitted[0]))
        return str(date.date().isoformat())
    elif splitted[1].lower() in ['day', 'days', 'd']:
        date = start_date - relativedelta(days=int(splitted[0]))
        return str(date.isoformat())
    elif splitted[1].lower() in ['wk', 'wks', 'week', 'weeks', 'w']:
        date = start_date - relativedelta(weeks=int(splitted[0]))
        return str(date.isoformat())
    elif splitted[1].lower() in ['mon', 'mons', 'month', 'months', 'm']:
        date = start_date - relativedelta(months=int(splitted[0]))
        return str(date.isoformat())
    elif splitted[1].lower() in ['yrs', 'yr', 'years', 'year', 'y']:
        date = start_date - relativedelta(years=int(splitted[0]))
        return str(date.isoformat())
    else:
        return "Wrong Argument format:" + splitted
    

def waiting_time(within_text):
    if splitted[1].lower() in ['hour', 'hours', 'hr', 'hrs', 'h']:
        relativedelta(hours=int(splitted[0]))
        return str(date.date().isoformat())
    elif splitted[1].lower() in ['day', 'days', 'd']:
        relativedelta(days=int(splitted[0]))
        return str(date.isoformat())
    elif splitted[1].lower() in ['wk', 'wks', 'week', 'weeks', 'w']:
        relativedelta(weeks=int(splitted[0]))
        return str(date.isoformat())
    elif splitted[1].lower() in ['mon', 'mons', 'month', 'months', 'm']:
        relativedelta(months=int(splitted[0]))
        return str(date.isoformat())
    elif splitted[1].lower() in ['yrs', 'yr', 'years', 'year', 'y']:
        relativedelta(years=int(splitted[0]))
        return str(date.isoformat())
    else:
        return "Wrong Argument format"