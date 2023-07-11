import suncalc
import datetime
import grape

data_dir = 'wwv_data/'
g = grape.Grape(data_dir + 'wwv_october_2022/2022-10-03T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)

print(g.suntimes)
print()

for i in range(0, int(len(g.time)/10)):
    print(g.time[i])
    print(g.sunpos[i])
