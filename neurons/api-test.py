import requests



# Define the base URL for the API
base_url = "https://gamma-api.polymarket.com/markets"

# Define the parameters for the API request
params = {
    # "limit": 1,
    "start_date_min": "2024-08-01T00:00:00Z"  # Adjust the limit as needed
}

# Make the API request
response = requests.get(base_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    # print(data)
    # Extract the titles from the response
    titles = [event['question'] for event in data if 'question' in event]
    dates = [event['endDate'] for event in data if 'endDate' in event]
    info = list(zip(titles, dates))
    
    # # Print the titles
    print(info)
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

