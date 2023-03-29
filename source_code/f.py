a =[0,67,9,67,67]
taget = 76

for x in range(0,len(a)):
    for y in range(x+1, len(a)):
        if a[x]+a[y]==taget:
            print(x,y)
        
