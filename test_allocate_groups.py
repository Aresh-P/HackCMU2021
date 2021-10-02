from allocate_groups import *

schedules = [[0, 0, 1], [0, 1, 1]]
sizes = [{"mingroup": 2, "maxgroup": 2}, {"mingroup": 2, "maxgroup": 2}]
courses = 2
alpha = 10
beta = 1
iterations = 100
print(allocate_groups(schedules, sizes, courses, alpha, beta, iterations))
