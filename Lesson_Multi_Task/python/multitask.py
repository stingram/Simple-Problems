def multitask(tasks, cooldown):
    t = -1
    tdict= {}
    sol = []
    for i in range(len(tasks)):
        if tasks[i] in tdict:
            time_ran = tdict[tasks[i]]
            if t > time_ran + cooldown:
                sol.append(tasks[i])
                tdict[tasks[i]] = t
            else:
                while(t<time_ran+cooldown):
                    sol.append("-")
                    t+=1
                sol.append(tasks[i])
                
                # updat time and tdict
                t += 1
                tdict[tasks[i]] = t
                
        else:
            sol.append(tasks[i])
            
            # update time and dict
            t = t + 1
            tdict[tasks[i]] = t
            
    return sol

def multitask_v2(tasks, cooldown):
    tdict = {}
    curr_time = 0
    
    for task in tasks:
        if task in tdict:
            if curr_time - tdict[task] <= cooldown:
                #add cooldown
                curr_time = cooldown + tdict[task] + 1
        tdict[task] = curr_time
        curr_time += 1
    return curr_time

my_list = [1,2,3,4,5,6,2,1]
cooldown = 2

print(multitask(my_list, cooldown))
print(multitask_v2(my_list, cooldown))