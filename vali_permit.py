import bittensor as bt

user_uid = int(input('Your uid:'))
user_wallet = input('Your wallet name:')
user_hotkey = input('hotkey:')
subnet = bt.metagraph(6)
st = bt.subtensor('finney')
top_64_stake = subnet.S.sort()[0][-64:].tolist()
wallet = bt.wallet(name=user_wallet,hotkey=user_hotkey)

print (f'Current requirement for validator permits based on the top 64 stake stands at {min(top_64_stake)} tao')
print(f'Your UID: {user_uid} validator permit: ', subnet.validator_permit[user_uid])
