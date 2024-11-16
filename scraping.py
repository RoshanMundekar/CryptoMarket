# -*- coding: utf-8 -*-

import csv
from datetime import datetime
import requests

response = requests.get('https://api.coingecko.com/api/v3/coins/lota/market_chart',
                        params={'vs_currency': 'usd', 'days': '450'})

if response.status_code == 200:
    data = response.json()
    prices = data['prices']
else:
    print('Error occurred while fetching data:', response.status_code)
    exit()

with open('lota.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Start', 'Close'])
    
    for entry in prices:
        timestamp = entry[0] / 1000  # Convert milliseconds to seconds
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')  # Convert timestamp to date
        start_price = entry[1]
        close_price = entry[1]
        writer.writerow([date, start_price, close_price])
