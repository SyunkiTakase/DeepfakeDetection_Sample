import requests

def line_notify(message):
    line_notify_token = 'y2uDm239QbtuPU1XygtWPI8WwINHxhS1VEQypXHznV8'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)
