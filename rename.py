import os

archivos = os.listdir()

for a in archivos:
    num = a.replace('img_','')
    num = int(num.replace('.png',''))
    os.rename(a,f"img_{num:03}.png")