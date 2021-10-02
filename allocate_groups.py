import math
import random

def probability_distribution(schedules, groups, sizes, courses, alpha, beta):
    """
    schedules is of the form:
    For each student (list):
      For each time (list):
        What subject are they working on (specified as int, -1 means not studying)

    groups is of the form:
    For each student (list):
      For each time (list):
        What study group (int) are they assigned to. (-1 means not in any group)

    sizes is of the form:
    For each student (list):
      {"mingroup": min desired study group size, "maxgroup": max desired study group size}

    courses is the number of courses people are interested in.
    study groups per course = number of students = len(groups).

    cost function is alpha * group_size_penalty + beta * continuity_penalty,
    log of probabilities is proportional to e^(-cost function).

    Returns:
    For each student (list):
      For each time (list):
        For each study group (list):
          Normalized probability student should be assigned to that study group.

    For each student (list):
      For each time (list):
        For each study group (list):
          Non-normalized log of probability student should be assigned to that study group.
    """
    study_group_sizes = []
    for study_group in range(courses*len(groups)):
        study_group_counts = []
        for time in range(len(groups[0])):
            count = 0
            for student in range(len(groups)):
                if groups[student][time] == study_group:
                    count += 1
            study_group_counts.append(count)
        study_group_sizes.append(study_group_counts)
    
    probabilities = []
    log_raw = []
    for student in range(len(groups)):
        student_probabilities = []
        student_log_raw = []
        for time in range(len(groups[0])):
            log_allocation_probabilities = [] #not normalized!
            for study_group in range(courses*len(groups)):
                course = int(study_group / len(groups))
                if schedules[student][time] != course:
                    log_allocation_probabilities.append(-float("inf")) #log 0 = -inf
                else:
                    if groups[student][time] == study_group:
                        #study group size already takes into account this student's presence
                        potential_group_size = study_group_sizes[study_group][time]
                    else:
                        #study group size does not take into account this student's presence
                        potential_group_size = study_group_sizes[study_group][time] + 1

                    over_penalty = max(0, potential_group_size - sizes[student]["maxgroup"])
                    under_penalty = max(0, sizes[student]["mingroup"] - potential_group_size)
                    #only one of over_penalty and under_penalty should be non-zero at a time.
                    group_size_allocation_penalty = over_penalty + under_penalty

                    continuity_penalty = 0
                    if time > 0:
                        if study_group != groups[student][time - 1]:
                            continuity_penalty += 1
                    if time < len(groups[0]) - 1:
                        if study_group != groups[student][time + 1]:
                            continuity_penalty += 1

                    overall_penalty = alpha * group_size_allocation_penalty + beta * continuity_penalty
                    log_allocation_probabilities.append(-overall_penalty)

            student_log_raw.append(log_allocation_probabilities)

            max_log_probability = max(log_allocation_probabilities)
            if max_log_probability == -float("inf"):
                #schedules[student][time] is not any of the courses,
                #so it must be -1, i.e. the student isn't studying.
                #The probabilities should all be zero.
                normalized_probabilities = [0 for log_probability in log_allocation_probabilities]
            else:
                #student is assigned to a course.
                non_normalized_probabilities = [math.exp(log_probability - max_log_probability) for log_probability in log_allocation_probabilities]
                normalization = sum(non_normalized_probabilities)
                normalized_probabilities = [non_normalized_probability / normalization for non_normalized_probability in non_normalized_probabilities]
            student_probabilities.append(normalized_probabilities)
        log_raw.append(student_log_raw)
        probabilities.append(student_probabilities)
    return (probabilities, log_raw)


def allocate_groups(schedules, sizes, courses, alpha, beta, iterations):
    """
    schedules is of the form:
    For each student (list):
      For each time (list):
        What subject are they working on (specified as int, -1 means not studying)

    courses is the number of courses people are interested in.
    study groups per course = number of students = len(groups).

    cost function is alpha * group_size_penalty + beta * continuity_penalty,
    log of probabilities is proportional to e^(-cost function).

    iterations is the number of times to run the reallocation step.

    Returns:
    For each student (list):
      For each time (list):
        What study group (int) they are assigned to. (-1 means not in any group)
    """
    
    groups = []
    for student in range(len(schedules)):
        student_groups = []
        for time in range(len(schedules[0])):
            course = schedules[student][time]
            if course == -1:
                student_groups.append(-1)
            else:
                study_group = student + course * len(schedules)
                student_groups.append(study_group)
        groups.append(student_groups)

    costs_over_time = []
    groups_over_time = []
    for iteration in range(iterations):
        probability_output = probability_distribution(schedules, groups, sizes, courses, alpha, beta)
        probabilities = probability_output[0]
        log_raw = probability_output[1]

        total_cost = 0
        for student in range(len(schedules)):
            for time in range(len(schedules[0])):
                if schedules[student][time] == -1:
                    groups[student][time] = -1
                    #total_cost should remain untouched.
                else:
                    CDF_desired = random.random()
                    CDF = 0
                    #use default if probabilities don't work.
                    selected_group = student + schedules[student][time] * len(schedules)
                    for i in range(len(probabilities[student][time])):
                        CDF += probabilities[student][time][i]
                        if CDF >= CDF_desired:
                            selected_group = i
                            break
                    groups[student][time] = selected_group
                    total_cost -= log_raw[student][time][selected_group] #value is negative, so subtract to add cost.
        
        costs_over_time.append(total_cost)
        groups_over_time.append(groups)

    min_cost = min(costs_over_time)
    best_iteration = costs_over_time.index(min_cost)
    return groups_over_time[best_iteration]
