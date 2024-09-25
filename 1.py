# 101 000 101
def fangge(nums):
    day = 0
    if nums is None and all(nums)==1:
        return -1
    while all(nums)!=1:
        z = []
        for i in range(len(nums)):
            if nums[i]==1:
                z.append(i)
        for s,num in enumerate(z):
            if s%3==1:
                nums[s+1]=1
                nums[s+3]=1
                nums[s-3]=1
        day +=1
    return day

day = fangge([1,0,1,0,0,0,1,0,1])
print(day)

