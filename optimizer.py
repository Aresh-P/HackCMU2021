from cost import cost


def optimizer(user_requirements):
    """
    Input:
    Each user requirement consists of:
      1.) A sorted list of intervals of times the user is free
          [{"start": time1, "stop": time2}, ...]
          times are measured in hours from the current time,
          so tomorrow is 24, two days from now is 48, etcetera.
      2.) A list of deadlines and study requirements for each deadline
          [{"time": time1, "course": course1, "hours": hours1}, ...]
      3.) A range of acceptable study group sizes
          {"mingroup": min study group size, "maxgroup": max study group size}
    
    Output:
    A list for each study group meeting:
      1.) {"start": time1,
           "stop": time2,
           "course": course,
           "students": {student1, student2, ...}}
    """
    pass
