text = "X-DSPAM-Confidence:    0.8475"
zero_pos = text.find('0')
num = text[zero_pos:]
float_num =float(num)
print(float_num)
