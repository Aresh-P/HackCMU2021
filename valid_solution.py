def valid_solution(user_requirements, proposed_solution):
    """
    This is a specification function and is not intended to be fast.

    arguments are of the form specified in optimizer.py.
    
    Requirements:
      1.) No user should be assigned to a study group while they are busy.
      2.) No user should be assigned to a study group for a subject which they are not working on.
    """
    for proposed_meeting in proposed_solution:
        for user in user_requirements:

