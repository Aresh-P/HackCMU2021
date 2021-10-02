def groups_to_output(groups):
    """
    groups is of the form:
    For each student (list):
        For each time (list):
            The study group number (int) the student is assigned to at that time (-1 for no study group).

    Returns:
    For each student (list):
        For each study session (list sorted by time):
    {"start": time, "stop": time, "study group number": int, "subject": int}
    stop time is the first time interval after the meeting is over! It is not included in the meeting!
    """
    sessions = []
    for student in range(len(groups)):
        student_sessions = []
        time = 0
        ongoing_session = None
        while time < len(groups[0]):
            if ongoing_session == None:
                if groups[student][time] != -1:
                    ongoing_session = {}
                    ongoing_session["start"] = time
                    ongoing_session["study group number"] = groups[student][time]
                    ongoing_session["subject"] = int(groups[student][time] / len(groups))
            else:
                if groups[student][time] != ongoing_session["study group number"]:
                    ongoing_session["stop"] = time
                    student_sessions.append(ongoing_session)
                    if groups[student][time] == -1:
                        ongoing_session = None
                    else:
                        ongoing_session = {}
                        ongoing_session["start"] = time
                        ongoing_session["study group number"] = groups[student][time]
                        ongoing_session["subject"] = int(groups[student][time] / len(groups))
            time += 1

        if ongoing_session != None:
            ongoing_session["stop"] = len(groups[0])
            student_sessions.append(ongoing_session)
            ongoing_session = None

        sessions.append(student_sessions)
    return sessions
