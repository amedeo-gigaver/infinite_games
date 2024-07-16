
from time import sleep
import bittensor as bt


def some():

    sixth_hotkey = bt.wallet(name='mymain', hotkey='sixth')
    sec_hotkey = bt.wallet(name='mymain', hotkey='second')
    s = bt.subtensor()
    # m = s.metagraph(6)

    sleep(2)
    print('Registering..')
    bt.debug(True)
    bt.trace(True)
    success = False
    while not success:
        try:
            success = s.burned_register(sixth_hotkey, 6)
        except Exception as e:
            print(e)
        sleep(2)
    success = False
    print(f"SN {6} Register result of {sixth_hotkey.hotkey.ss58_address}: {success}")
    while not success:
        try:
            success = s.burned_register(sec_hotkey, 6)
        except Exception as e:
            print(e)
        sleep(2)
    print(f"SN {6} Register result of {sec_hotkey.hotkey.ss58_address}: {success}")


if __name__ == '__main__':
    some()