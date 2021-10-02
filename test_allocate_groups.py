from allocate_groups import *
from groups_to_output import *

schedules = [[0], [0]]
sizes = [{"mingroup": 2, "maxgroup": 2}, {"mingroup": 2, "maxgroup": 2}]
courses = 2
alpha = 100
beta = 1
iterations = 100
groups = allocate_groups(schedules, sizes, courses, alpha, beta, iterations)
print(groups)
output = groups_to_output(groups)
print(output)
