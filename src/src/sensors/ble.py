b=['0x55', '0x61', '0x3e', '0x1', '0x30', '0x0', '0x2', '0x8', '0xff', '0xff', '0x8', '0x0', '0x0', '0x0', '0x2a', '0x0', '0xe4', '0xf8', '0xbf', '0xe2']
hex_int = [int(i, base=16) for i in b]
bytearray_obj = bytearray(hex_int)
print([str(i) for i in bytearray_obj])
print([str(hex(i)).split('x')[1] for i in bytearray_obj])
print(int('191', base=16))