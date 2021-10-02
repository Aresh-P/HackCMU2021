def optimizer(user_requirements):
    """
    Input:
    Each user requirement consists of:
      1.) A list of intervals of times the user is free
          [(time1, time2), (time3, time4), ...]
          times are measured in hours from the current time,
          so tomorrow is 24, two days from now is 48, etcetera.
      2.) A list of deadlines and study requirements for each deadline
          [(time1, course1, hours1), (time2, course2, hours2), ...]
      3.) A range of acceptable study group sizes
          (min study group size, max study group size)
    
    Output:
    For each study group meeting:
      1.) [time1, time2, course, [student1, student2, ...]]
    """

