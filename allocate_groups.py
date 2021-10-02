import math

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
          Probability student should be assigned to that study group.
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
    for student in range(len(groups)):
        student_probabilities = []
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
        probabilities.append(student_probabilities)
    return probabilities


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

    for iteration in range(iterations):

