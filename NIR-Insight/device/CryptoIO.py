from cryptoauthlib import *
from cryptoauthlib.device import *
from lib.external.common import *
import time

# Slot 4 IO Encryption key
ENC_KEY = bytearray([
    0x37, 0x80, 0xe6, 0x3d, 0x49, 0x68, 0xad, 0xe5,
    0xd8, 0x22, 0xc0, 0x13, 0xfc, 0xc3, 0x23, 0x84,
    0x5d, 0x1b, 0x56, 0x9f, 0xe7, 0x05, 0xb6, 0x00,
    0x06, 0xfe, 0xec, 0x14, 0x5a, 0x0d, 0xb1, 0xe3
])

def write(data, iface='i2c', device='ecc'):
    ATCA_SUCCESS = 0x00

    # Loading cryptoauthlib(python specific)
    load_cryptoauthlib()

    # Get the target default config
    cfg = eval('cfg_at{}a_{}_default()'.format(atca_names_map.get(device), atca_names_map.get(iface)))

    # Basic Raspberry Pi I2C check
    if 'i2c' == iface and check_if_rpi():
        cfg.cfg.atcai2c.bus = 1

    # Initialize the stack
    assert atcab_init(cfg) == ATCA_SUCCESS

    # Check device type
    info = bytearray(4)
    assert atcab_info(info) == ATCA_SUCCESS
    dev_name = get_device_name(info)
    dev_type = get_device_type_id(dev_name)

    # Reinitialize if the device type doesn't match the default
    if dev_type != cfg.devtype:
        cfg.dev_type = dev_type
        assert atcab_release() == ATCA_SUCCESS
        time.sleep(1)
        assert atcab_init(cfg) == ATCA_SUCCESS

    # Read the config to find some setup values
    config_data = bytearray(128)
    assert ATCA_SUCCESS == atcab_read_config_zone(config_data)
    config = Atecc508aConfig.from_buffer(config_data)

    # Find the write key slot for the encrypted write slot
    write_key_slot = config.SlotConfig[8].WriteKey

    # Writing IO protection key. This key is used as IO encryption key.
    assert atcab_write_zone(2, write_key_slot, 0, 0, ENC_KEY, 32) == ATCA_SUCCESS

    for i in range(len(data)):
        assert atcab_write_enc(8, i, data[i], ENC_KEY, write_key_slot) == ATCA_SUCCESS

    # Free the library
    atcab_release()


def read(length, iface='i2c', device='ecc'):
    ATCA_SUCCESS = 0x00

    # Loading cryptoauthlib(python specific)
    load_cryptoauthlib()

    # Get the target default config
    cfg = eval('cfg_at{}a_{}_default()'.format(atca_names_map.get(device), atca_names_map.get(iface)))

    # Basic Raspberry Pi I2C check
    if 'i2c' == iface and check_if_rpi():
        cfg.cfg.atcai2c.bus = 1

    # Initialize the stack
    assert atcab_init(cfg) == ATCA_SUCCESS

    # Check device type
    info = bytearray(4)
    assert atcab_info(info) == ATCA_SUCCESS
    dev_name = get_device_name(info)
    dev_type = get_device_type_id(dev_name)

    # Reinitialize if the device type doesn't match the default
    if dev_type != cfg.devtype:
        cfg.dev_type = dev_type
        assert atcab_release() == ATCA_SUCCESS
        time.sleep(1)
        assert atcab_init(cfg) == ATCA_SUCCESS

    # Read the config to find some setup values
    config_data = bytearray(128)
    assert ATCA_SUCCESS == atcab_read_config_zone(config_data)
    config = Atecc508aConfig.from_buffer(config_data)

    # Find the write key slot for the encrypted write slot
    write_key_slot = config.SlotConfig[8].WriteKey

    # Writing IO protection key. This key is used as IO encryption key.
    assert atcab_write_zone(2, write_key_slot, 0, 0, ENC_KEY, 32) == ATCA_SUCCESS

    read_data = []
    for i in range(length):
        read_data.append(bytearray(32))
    for i in range(length):
        assert atcab_read_enc(8, i, read_data[i], ENC_KEY, write_key_slot) == ATCA_SUCCESS

    # Free the library
    atcab_release()
    return read_data